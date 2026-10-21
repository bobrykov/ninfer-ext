// The direct n-gram row reader must retain the 4 KiB pages it reads, up to an explicit page budget,
// so a later call touching the same rows is served from RAM instead of re-reading the volume; a
// zero budget disables retention and re-reads everything. Retention must never change the bytes a
// range reads, including when the budget is smaller than one call's working set and pages churn.
// A background prefetch fills the same retained pages; the next read joins it, finds its pages
// resident, and a reader destroyed with a prefetch in flight must join it cleanly.
// The synthetic table spans two volume files at unaligned offsets, so rows straddle 4 KiB pages and
// the second segment starts mid-page.
#include "models/qwen3_5/execution/ple_rows.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <span>
#include <vector>

using namespace ninfer;
using namespace ninfer::models::qwen3_5::execution;

namespace {

constexpr std::uint32_t kRows        = 1500;
constexpr std::uint64_t kRowBytes    = 16; // FP8 row: one byte per value
constexpr std::uint64_t kScaleBytes  = 2;
constexpr std::uint64_t kScalePlane  = (kRows * kRowBytes + 255) / 256 * 256;
constexpr std::uint64_t kObjectBytes = kScalePlane + kRows * kScaleBytes;
constexpr std::uint64_t kSplit       = 700 * kRowBytes; // row boundary inside the first file

// Spread across both files and across 4 KiB pages, since consecutive rows share pages.
constexpr std::uint32_t kProbeRows[]    = {0, 300, 600, 900, 1200, 1440};
constexpr std::uint32_t kNeighborRows[] = {1, 299, 601, 901, 1201, 1439};
// Rows whose 16 code bytes straddle a 4 KiB page boundary: 1234 + 178*16 = 4082 in a.bin and
// 777 + 207*16 = 4089 in b.bin.
constexpr std::uint32_t kStraddlingRows[] = {178, 207};

struct Volume {
    std::filesystem::path dir;
    std::vector<std::byte> object; // original bytes; the oracle every unmutated read must return
};

void write_file(const std::filesystem::path& path, std::uint64_t leading, const std::byte* data,
                std::size_t bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    const std::vector<char> pad(leading, 'x');
    out.write(pad.data(), static_cast<std::streamsize>(pad.size()));
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    // Trailing bytes so the last direct page read is not short.
    const std::vector<char> tail(8192, 'y');
    out.write(tail.data(), static_cast<std::streamsize>(tail.size()));
}

Volume make_volume() {
    Volume volume;
    volume.object.resize(kObjectBytes);
    std::mt19937 rng(20260820);
    for (auto& byte : volume.object) { byte = std::byte{static_cast<std::uint8_t>(rng())}; }

    volume.dir = std::filesystem::current_path() / "ple_rows_test_volumes";
    std::filesystem::create_directories(volume.dir);
    write_file(volume.dir / "a.bin", 1234, volume.object.data(), kSplit);
    write_file(volume.dir / "b.bin", 777, volume.object.data() + kSplit, kObjectBytes - kSplit);
    return volume;
}

void rewrite_files(const Volume& volume, const std::vector<std::byte>& object) {
    write_file(volume.dir / "a.bin", 1234, object.data(), kSplit);
    write_file(volume.dir / "b.bin", 777, object.data() + kSplit, kObjectBytes - kSplit);
}

// Object image the ranges of `rows` must read.
std::vector<PleRowReader::Range> ranges_of(std::span<const std::uint32_t> rows) {
    std::vector<PleRowReader::Range> ranges;
    ranges.reserve(rows.size() * 2);
    for (const auto row : rows) {
        ranges.push_back({row * kRowBytes, 16});
        ranges.push_back({kScalePlane + row * kScaleBytes, 2});
    }
    return ranges;
}

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

int matches(const PleRowReader& reader, std::span<const PleRowReader::Range> ranges,
            const std::vector<std::byte>& oracle, const char* message) {
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        if (std::memcmp(reader.data(i), oracle.data() + ranges[i].object_offset, ranges[i].bytes) !=
            0) {
            std::cerr << message << " (range " << i << ")\n";
            return 1;
        }
    }
    return 0;
}

std::span<const artifact::MappedObjectSegment> segments_of(const Volume& volume) {
    static thread_local std::vector<artifact::MappedObjectSegment> segments;
    segments = {
        {0, kSplit, volume.object.data(), volume.dir / "a.bin", 1234},
        {kSplit, kObjectBytes - kSplit, volume.object.data() + kSplit, volume.dir / "b.bin", 777},
    };
    return segments;
}

int retained_pages_are_not_reread(const Volume& volume) {
    const auto rows    = std::span<const std::uint32_t>(kProbeRows);
    const auto ranges  = ranges_of(rows);
    const auto flipped = [&] {
        std::vector<std::byte> object = volume.object;
        for (const auto row : rows) {
            for (std::uint64_t j = 0; j < kRowBytes; ++j) { object[row * kRowBytes + j] ^= std::byte{0xFF}; }
        }
        return object;
    }();

    // Every page the probe rows touch fits the budget, so the second read must come from RAM.
    PleRowReader reader(segments_of(volume), 64);
    reader.read(ranges);
    int failures = matches(reader, ranges, volume.object, "first read is not the stored object");

    rewrite_files(volume, flipped);
    reader.read(ranges);
    failures += matches(reader, ranges, volume.object,
                        "retained pages were re-read after the volume changed");
    // A fresh reader with the same budget must see the new bytes, proving the files really changed.
    PleRowReader fresh(segments_of(volume), 64);
    fresh.read(ranges);
    failures += matches(fresh, ranges, flipped, "a fresh reader did not see the rewritten volume");
    // Leave the volume holding the oracle bytes for the tests that follow.
    rewrite_files(volume, volume.object);
    return failures;
}

int zero_budget_rereads(const Volume& volume) {
    const auto ranges = ranges_of(std::span<const std::uint32_t>(kProbeRows));
    PleRowReader reader(segments_of(volume), 0);
    reader.read(ranges);
    const auto original = matches(reader, ranges, volume.object, "uncached first read differs");

    std::vector<std::byte> object = volume.object;
    for (const auto row : kProbeRows) {
        for (std::uint64_t j = 0; j < kRowBytes; ++j) { object[row * kRowBytes + j] ^= std::byte{0x55}; }
    }
    rewrite_files(volume, object);
    reader.read(ranges);
    return original + matches(reader, ranges, object, "zero budget did not re-read the volume") +
           // Restore the oracle bytes: later tests compare reads against `volume.object`.
           [&] {
               rewrite_files(volume, volume.object);
               return 0;
           }();
}

// Many overlapping calls with a budget that holds every page they touch: every call must agree with
// the object image, whatever slots and reuse the hash order produces.
int stable_across_overlapping_calls(const Volume& volume) {
    std::vector<std::uint32_t> universe;
    for (const auto row : kProbeRows) { universe.push_back(row); }
    for (const auto row : kNeighborRows) { universe.push_back(row); }
    for (const auto row : kStraddlingRows) { universe.push_back(row); }
    const auto segments = segments_of(volume);

    PleRowReader retained(segments, 128);
    PleRowReader uncached(segments, 0);
    std::mt19937 rng(77);
    int failures = 0;
    for (int call = 0; call < 24; ++call) {
        std::vector<std::uint32_t> rows(universe.begin() + rng() % 4, universe.end() - rng() % 4);
        const auto ranges = ranges_of(rows);
        retained.read(ranges);
        uncached.read(ranges);
        for (std::size_t i = 0; i < ranges.size(); ++i) {
            if (std::memcmp(retained.data(i), uncached.data(i), ranges[i].bytes) != 0) {
                std::cerr << "retained call " << call << " range " << i
                          << " differs from the uncached bytes\n";
                failures = 1;
                break;
            }
        }
        if (failures != 0) { break; }
    }
    return failures;
}

// A budget smaller than one call's working set: pages evict and overflow while the call proceeds,
// and the bytes must still be exact.
int capacity_pressure_is_correct(const Volume& volume) {
    std::vector<std::uint32_t> universe;
    for (const auto row : kProbeRows) { universe.push_back(row); }
    for (const auto row : kNeighborRows) { universe.push_back(row); }
    for (const auto row : kStraddlingRows) { universe.push_back(row); }
    const auto segments = segments_of(volume);

    PleRowReader pressured(segments, 4);
    PleRowReader uncached(segments, 0);
    int failures = 0;
    for (int call = 0; call < 8 && failures == 0; ++call) {
        const auto ranges = ranges_of(universe);
        pressured.read(ranges);
        uncached.read(ranges);
        for (std::size_t i = 0; i < ranges.size(); ++i) {
            if (std::memcmp(pressured.data(i), uncached.data(i), ranges[i].bytes) != 0) {
                std::cerr << "capacity-pressured call " << call << " range " << i << " is wrong\n";
                failures = 1;
                break;
            }
        }
    }
    return failures;
}

// A background prefetch fills the retained pages without producing row data: the next read joins
// it, finds its pages resident (proven by the hit counters), never re-reads a rewritten volume,
// and destroying a reader with a prefetch in flight joins it without disturbing later readers.
int prefetch_populates_and_joins(const Volume& volume) {
    const auto rows     = std::span<const std::uint32_t>(kProbeRows);
    const auto ranges   = ranges_of(rows);
    const auto segments = segments_of(volume);
    int failures        = 0;

    {
        PleRowReader reader(segments, 64);
        const auto before = reader.stats();
        reader.prefetch(ranges);
        reader.read(ranges); // joins the prefetch; every distinct page must now be a hit
        const auto after = reader.stats();
        failures += check(after.calls == before.calls + 2,
                          "prefetch and read did not both register as calls");
        failures += check(after.hits > before.hits,
                          "pages populated by the prefetch were not hits at read time");
        failures += matches(reader, ranges, volume.object, "read after prefetch is not the object");
    }

    // Resident pages must not leak a rewritten volume through either call path.
    {
        PleRowReader reader(segments, 64);
        reader.read(ranges);
        const auto flipped = [&] {
            std::vector<std::byte> object = volume.object;
            for (const auto row : rows) {
                for (std::uint64_t j = 0; j < kRowBytes; ++j) {
                    object[row * kRowBytes + j] ^= std::byte{0xFF};
                }
            }
            return object;
        }();
        rewrite_files(volume, flipped);
        reader.prefetch(ranges); // advisory; all pages resident, so nothing re-reads
        reader.read(ranges);
        failures += matches(reader, ranges, volume.object,
                            "prefetch re-read a rewritten volume instead of retaining");
        rewrite_files(volume, volume.object);
    }

    // Destruction with a prefetch in flight joins it; a later reader is unaffected.
    {
        {
            PleRowReader doomed(segments, 64);
            doomed.prefetch(ranges);
        }
        PleRowReader after(segments, 64);
        after.read(ranges);
        failures += matches(after, ranges, volume.object,
                            "read after a destroyed prefetching reader is wrong");
    }
    return failures;
}

} // namespace

int main() {
    const Volume volume = make_volume();
    int failures        = 0;
    try {
        failures += retained_pages_are_not_reread(volume);
        failures += zero_budget_rereads(volume);
        failures += stable_across_overlapping_calls(volume);
        failures += capacity_pressure_is_correct(volume);
        failures += prefetch_populates_and_joins(volume);
    } catch (const std::runtime_error& error) {
        // O_DIRECT is unsupported on some filesystems (tmpfs); that is an environment limit.
        std::cerr << "SKIP: " << error.what() << '\n';
        std::filesystem::remove_all(volume.dir);
        return 77;
    }
    std::filesystem::remove_all(volume.dir);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " PLE row reader page retention\n";
    return failures == 0 ? 0 : 1;
}
