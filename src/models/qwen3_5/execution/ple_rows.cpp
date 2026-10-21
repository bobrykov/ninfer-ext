#include "models/qwen3_5/execution/ple_rows.h"

#include "core/host_worker_pool.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen3_5::execution {
namespace {

constexpr std::uint64_t kPage       = PleRowReader::kPage;
constexpr std::uint32_t kReaders    = 16; // outstanding direct reads
constexpr std::size_t kPagesPerTask = 8;

} // namespace

struct PleRowReader::File {
    std::filesystem::path path;
    int fd = -1;

    explicit File(std::filesystem::path p) : path(std::move(p)) {
        fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (fd < 0) {
            throw std::runtime_error(
                path.string() + ": cannot open for direct n-gram reads: " + std::strerror(errno));
        }
    }

    ~File() {
        if (fd >= 0) { ::close(fd); }
    }

    void read_page(std::uint64_t offset, std::byte* destination) const {
        std::size_t done = 0;
        while (done < kPage) {
            const ssize_t n =
                ::pread(fd, destination + done, kPage - done, static_cast<off_t>(offset + done));
            if (n < 0 && errno == EINTR) { continue; }
            if (n < 0) {
                throw std::runtime_error(path.string() +
                                         ": direct n-gram read failed: " + std::strerror(errno));
            }
            if (n == 0) {
                // A short final page: the rest of the page lies past the end of the volume.
                std::memset(destination + done, 0, kPage - done);
                return;
            }
            done += static_cast<std::size_t>(n);
        }
    }
};

PleRowReader::PleRowReader(std::span<const artifact::MappedObjectSegment> segments,
                           std::size_t retained_pages)
    : segments_(segments.begin(), segments.end()),
      workers_(std::make_unique<HostWorkerPool>(kReaders, 4096)),
      retained_pages_(retained_pages) {
    if (segments_.empty()) { throw std::invalid_argument("n-gram reader has no segments"); }
    for (const auto& segment : segments_) {
        const auto it =
            std::ranges::find_if(files_, [&](const auto& f) { return f->path == segment.file; });
        if (it != files_.end()) {
            segment_file_.push_back(static_cast<std::uint32_t>(it - files_.begin()));
        } else {
            segment_file_.push_back(static_cast<std::uint32_t>(files_.size()));
            files_.push_back(std::make_unique<File>(segment.file));
        }
    }
    if (retained_pages_ != 0) {
        retained_ = static_cast<std::byte*>(
            std::aligned_alloc(kPage, retained_pages_ * kPage));
        if (retained_ == nullptr) {
            retained_pages_ = 0;
            throw std::bad_alloc();
        }
        stat_logged_ = std::chrono::steady_clock::now();
        retained_page_.resize(retained_pages_);
        lru_prev_.resize(retained_pages_, kNoSlot);
        lru_next_.resize(retained_pages_, kNoSlot);
        free_slots_.resize(retained_pages_);
        for (std::size_t slot = 0; slot < retained_pages_; ++slot) {
            free_slots_[slot] = retained_pages_ - 1 - slot;
        }
    }
}

PleRowReader::~PleRowReader() {
    join_pending();
    workers_.reset();
    std::free(staging_);
    std::free(retained_);
}

PleRowReader::Page PleRowReader::page_of(std::uint64_t object_offset, std::uint32_t bytes,
                                         std::uint32_t& pages, std::uint32_t& within) const {
    const auto it =
        std::upper_bound(segments_.begin(), segments_.end(), object_offset,
                         [](std::uint64_t offset, const artifact::MappedObjectSegment& s) {
                             return offset < s.object_offset;
                         });
    if (it == segments_.begin()) { throw std::out_of_range("n-gram offset is outside the table"); }
    const auto& segment        = *std::prev(it);
    const std::uint64_t inside = object_offset - segment.object_offset;
    if (inside + bytes > segment.bytes) {
        throw std::out_of_range("n-gram range crosses a volume segment");
    }
    const std::uint64_t file_offset = segment.file_offset + inside;
    const std::uint64_t first       = file_offset / kPage * kPage;
    pages  = static_cast<std::uint32_t>((file_offset + bytes - first + kPage - 1) / kPage);
    within = static_cast<std::uint32_t>(file_offset - first);
    return Page{segment_file_[static_cast<std::size_t>(it - segments_.begin()) - 1], first};
}

void PleRowReader::reserve(std::size_t pages) {
    if (pages <= staging_pages_) { return; }
    std::free(staging_);
    staging_       = static_cast<std::byte*>(std::aligned_alloc(kPage, pages * kPage));
    staging_pages_ = staging_ != nullptr ? pages : 0;
    if (staging_ == nullptr) { throw std::bad_alloc(); }
}

void PleRowReader::lru_unlink(std::size_t slot) {
    const std::size_t prev = lru_prev_[slot], next = lru_next_[slot];
    if (prev != kNoSlot) { lru_next_[prev] = next; } else { lru_head_ = next; }
    if (next != kNoSlot) { lru_prev_[next] = prev; } else { lru_tail_ = prev; }
    lru_prev_[slot] = lru_next_[slot] = kNoSlot;
}

void PleRowReader::lru_push_front(std::size_t slot) {
    lru_prev_[slot] = kNoSlot;
    lru_next_[slot] = lru_head_;
    if (lru_head_ != kNoSlot) { lru_prev_[lru_head_] = slot; }
    lru_head_ = slot;
    if (lru_tail_ == kNoSlot) { lru_tail_ = slot; }
}

// A slot for a newly read page: a free one, or the least-recently-used page's once it is evicted.
// Every slot holding a page the current call needs sits at the list front (each was touched or
// inserted this call), so the tail is only ever a stale page — unless the call needs every slot,
// reported as kNoSlot so the caller overflows that page to staging instead.
std::size_t PleRowReader::acquire_slot(std::size_t resident_current) {
    if (!free_slots_.empty()) {
        const std::size_t slot = free_slots_.back();
        free_slots_.pop_back();
        return slot;
    }
    if (lru_tail_ == kNoSlot || resident_current >= retained_pages_) { return kNoSlot; }
    const std::size_t slot = lru_tail_;
    lru_unlink(slot);
    retained_slot_of_.erase(retained_page_[slot]);
    return slot;
}

void PleRowReader::join_pending() {
    if (prefetch_pending_.valid()) {
        // get(), not wait(): a prefetch that died mid-read must rethrow before its half-filled
        // slots are ever served as hits.
        prefetch_pending_.get();
        prefetch_pending_ = {};
    }
}

// Classifies and fills every distinct page the ranges touch. Worker reads go straight into their
// final home: a retained slot for pages already held or newly claimed, staging for the overflow
// once every slot holds a page this call needs. Pages already retained are not read at all.
void PleRowReader::populate_pages(std::span<const Range> ranges) {
    pages_.clear();
    call_placements_.assign(ranges.size(), Placement{});
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        std::uint32_t count = 0, within = 0;
        const Page first = page_of(ranges[i].object_offset, ranges[i].bytes, count, within);
        call_placements_[i] = {first, within, count};
        for (std::uint32_t p = 0; p < count; ++p) {
            pages_.push_back(Page{first.file, first.offset + p * kPage});
        }
    }
    std::sort(pages_.begin(), pages_.end());
    pages_.erase(std::unique(pages_.begin(), pages_.end()), pages_.end());

    page_home_.resize(pages_.size());
    page_missing_.resize(pages_.size());
    if (retained_pages_ == 0) {
        reserve(pages_.size());
        for (std::size_t p = 0; p < pages_.size(); ++p) {
            page_home_[p]    = staging_ + p * kPage;
            page_missing_[p] = true;
        }
    } else {
        std::size_t resident_current = 0;
        std::size_t overflow         = 0;
        std::size_t hits             = 0;
        for (std::size_t p = 0; p < pages_.size(); ++p) {
            if (const auto it = retained_slot_of_.find(pages_[p]); it != retained_slot_of_.end()) {
                lru_unlink(it->second);
                lru_push_front(it->second);
                page_home_[p]    = retained_ + it->second * kPage;
                page_missing_[p] = false;
                ++resident_current;
                ++hits;
                continue;
            }
            const std::size_t slot = acquire_slot(resident_current);
            if (slot == kNoSlot) {
                page_home_[p] = nullptr; // assigned to staging below
                page_missing_[p] = true;
                ++overflow;
                continue;
            }
            retained_page_[slot]        = pages_[p];
            retained_slot_of_[pages_[p]] = slot;
            lru_push_front(slot);
            page_home_[p]    = retained_ + slot * kPage;
            page_missing_[p] = true;
            ++resident_current;
        }
        reserve(overflow);
        std::size_t staged = 0;
        for (std::size_t p = 0; p < pages_.size(); ++p) {
            if (page_home_[p] == nullptr) {
                page_home_[p] = staging_ + staged * kPage;
                ++staged;
            }
        }
        const auto now = std::chrono::steady_clock::now();
        ++stat_calls_;
        stat_pages_ += pages_.size();
        stat_hits_ += hits;
        if (now - stat_logged_ >= std::chrono::seconds(10)) {
            std::fprintf(stderr,
                         "[ple-cache] calls=%llu pages=%llu hits=%llu rate=%.1f%% "
                         "resident=%zu/%zu pages\n",
                         static_cast<unsigned long long>(stat_calls_),
                         static_cast<unsigned long long>(stat_pages_),
                         static_cast<unsigned long long>(stat_hits_),
                         100.0 * static_cast<double>(stat_hits_) /
                             static_cast<double>(stat_pages_),
                         retained_pages_ - free_slots_.size(), retained_pages_);
            stat_logged_ = now;
        }
    }

    std::vector<std::future<void>> pending;
    for (std::size_t begin = 0; begin < pages_.size(); begin += kPagesPerTask) {
        const std::size_t end = std::min(pages_.size(), begin + kPagesPerTask);
        pending.push_back(workers_->submit([this, begin, end] {
            for (std::size_t p = begin; p < end; ++p) {
                if (!page_missing_[p]) { continue; }
                files_[pages_[p].file]->read_page(pages_[p].offset, page_home_[p]);
            }
        }));
    }
    for (auto& task : pending) { task.get(); }
}

void PleRowReader::prefetch(std::span<const Range> ranges) {
    if (retained_pages_ == 0) { return; }
    join_pending();
    const std::vector<Range> owned(ranges.begin(), ranges.end());
    prefetch_pending_ = workers_->submit([this, owned] { populate_pages(owned); });
}

void PleRowReader::read(std::span<const Range> ranges) {
    join_pending();
    populate_pages(ranges);

    // A range spanning a page boundary must expose contiguous bytes, and consecutive pages of one
    // range are adjacent in the sorted page list but not necessarily in retained slots. Copy each
    // straddling range's pages into one compact buffer; the staging path is already contiguous.
    located_override_.assign(ranges.size(), nullptr);
    if (retained_pages_ != 0) {
        std::size_t compact_bytes = 0;
        for (std::size_t i = 0; i < ranges.size(); ++i) {
            if (call_placements_[i].pages_spanned > 1) {
                compact_bytes += call_placements_[i].pages_spanned * kPage;
            }
        }
        if (compact_bytes != 0) {
            compact_.resize(compact_bytes);
            std::size_t filled = 0;
            for (std::size_t i = 0; i < ranges.size(); ++i) {
                const std::uint32_t spanned = call_placements_[i].pages_spanned;
                if (spanned <= 1) { continue; }
                const auto it = std::lower_bound(pages_.begin(), pages_.end(), call_placements_[i].page);
                const std::size_t first = static_cast<std::size_t>(it - pages_.begin());
                for (std::uint32_t p = 0; p < spanned; ++p) {
                    std::memcpy(compact_.data() + (filled + p) * kPage, page_home_[first + p], kPage);
                }
                located_override_[i] = compact_.data() + filled * kPage + call_placements_[i].within;
                filled += spanned;
            }
        }
    }

    // Consecutive pages of one range are adjacent in the sorted page list.
    located_.resize(ranges.size());
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        if (located_override_[i] != nullptr) {
            located_[i] = located_override_[i];
            continue;
        }
        const auto it = std::lower_bound(pages_.begin(), pages_.end(), call_placements_[i].page);
        located_[i] =
            page_home_[static_cast<std::size_t>(it - pages_.begin())] + call_placements_[i].within;
    }
}

} // namespace ninfer::models::qwen3_5::execution
