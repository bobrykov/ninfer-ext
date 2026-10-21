#pragma once

#include "artifact/materializer.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

namespace ninfer {
class HostWorkerPool;
}

namespace ninfer::models::qwen3_5::execution {

// Batched direct (O_DIRECT) reads of byte ranges of one file object, bypassing the page cache.
// Every call gathers the distinct 4 KiB pages its ranges touch and reads them concurrently. A
// nonzero page budget retains that many pages between calls — repeated n-grams are then served
// from RAM and only unseen pages reach the volume, least-recently-used pages evicted first.
// `data` addresses stay valid until the next call, which may evict or overwrite them.
class PleRowReader {
public:
    static constexpr std::uint64_t kPage = 4096;

    explicit PleRowReader(std::span<const artifact::MappedObjectSegment> segments,
                          std::size_t retained_pages = 0);
    ~PleRowReader();

    PleRowReader(const PleRowReader&)            = delete;
    PleRowReader& operator=(const PleRowReader&) = delete;

    struct Range {
        std::uint64_t object_offset = 0;
        std::uint32_t bytes         = 0;
    };

    // Reads every range; afterwards `data(i)` addresses range i. Ranges may repeat or overlap.
    void read(std::span<const Range> ranges);

    // Populates the retained pages for `ranges` on the worker pool without producing row data;
    // a following `read` of the same ranges then finds them resident. Joins any earlier pending
    // prefetch first, so reader state is only ever touched by one call at a time. A no-op when
    // retention is disabled. Exceptions propagate at the next join — a failed prefetch must not
    // leave silently half-filled pages behind.
    void prefetch(std::span<const Range> ranges);

    [[nodiscard]] bool retains() const noexcept { return retained_pages_ != 0; }

    struct Stats {
        std::uint64_t calls = 0;
        std::uint64_t pages = 0;
        std::uint64_t hits  = 0;
    };
    [[nodiscard]] Stats stats() const noexcept { return {stat_calls_, stat_pages_, stat_hits_}; }

    [[nodiscard]] const std::byte* data(std::size_t range) const { return located_.at(range); }

private:
    struct File;

    struct Page {
        std::uint32_t file   = 0;
        std::uint64_t offset = 0; // page-aligned file offset

        friend bool operator<(const Page& a, const Page& b) {
            return a.file != b.file ? a.file < b.file : a.offset < b.offset;
        }

        friend bool operator==(const Page&, const Page&) = default;
    };

    struct PageHash {
        std::size_t operator()(const Page& page) const noexcept {
            std::uint64_t x = page.offset * 0x9E3779B97F4A7C15ULL ^ (static_cast<std::uint64_t>(page.file) << 32);
            x ^= x >> 33;
            x *= 0xff51afd7ed558ccdULL;
            x ^= x >> 33;
            return static_cast<std::size_t>(x);
        }
    };

    [[nodiscard]] Page page_of(std::uint64_t object_offset, std::uint32_t bytes,
                               std::uint32_t& pages, std::uint32_t& within) const;
    void reserve(std::size_t pages);
    // Retained-page bookkeeping. Only the calling thread touches it, between the worker joins of
    // successive reads, so no locking is needed.
    [[nodiscard]] std::size_t acquire_slot(std::size_t resident_current);
    void lru_unlink(std::size_t slot);
    void lru_push_front(std::size_t slot);

    struct Placement {
        Page page;
        std::uint32_t within        = 0;
        std::uint32_t pages_spanned = 1;
    };
    // Classifies and fills every distinct page the ranges touch; leaves `call_placements_`,
    // `pages_`, `page_home_` and `page_missing_` describing the call. Shared by `read` (on the
    // calling thread) and `prefetch` (on a worker); the pending-prefetch join serializes the two.
    void populate_pages(std::span<const Range> ranges);
    void join_pending();

    static constexpr std::size_t kNoSlot = static_cast<std::size_t>(-1);

    std::vector<artifact::MappedObjectSegment> segments_;
    std::vector<std::unique_ptr<File>> files_;
    std::vector<std::uint32_t> segment_file_;
    std::unique_ptr<HostWorkerPool> workers_;
    std::byte* staging_        = nullptr;
    std::size_t staging_pages_ = 0;
    std::vector<Page> pages_;
    std::vector<const std::byte*> located_;
    // Retained pages: `retained_` holds the bytes, `retained_slot_of_` locates a page in it, and
    // `retained_page_` is the reverse map eviction needs. Free slots come from `free_slots_`
    // before the least-recently-used list tail recycles.
    std::size_t retained_pages_ = 0;
    std::byte* retained_        = nullptr;
    std::vector<Page> retained_page_;
    std::unordered_map<Page, std::size_t, PageHash> retained_slot_of_;
    std::vector<std::size_t> free_slots_;
    std::vector<std::size_t> lru_prev_, lru_next_; // per-slot links; kNoSlot ends both lists
    std::size_t lru_head_ = kNoSlot;
    std::size_t lru_tail_ = kNoSlot;
    std::vector<std::byte*> page_home_;  // per call: where each distinct page's bytes live
    std::vector<bool> page_missing_;    // per call: which distinct pages need reading
    std::vector<Placement> call_placements_; // per call: page placement of each range
    std::vector<std::byte> compact_;    // per call: contiguous bytes for page-straddling ranges
    std::vector<const std::byte*> located_override_; // per call: compacted homes of straddlers
    std::future<void> prefetch_pending_; // joined by read/prefetch/destructor before state is touched
    // Retention statistics, tracked only when retention is enabled and logged to stderr
    // periodically so a deployed run reports its own hit rate.
    std::uint64_t stat_calls_ = 0;
    std::uint64_t stat_pages_ = 0;
    std::uint64_t stat_hits_  = 0;
    std::chrono::steady_clock::time_point stat_logged_{};
};

} // namespace ninfer::models::qwen3_5::execution
