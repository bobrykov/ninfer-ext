#pragma once

#include "models/qwen3_5/config.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/execution/ple_rows.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

// Host gather of the Qwen4Exp n-gram embedding. The table stays file-mapped: every token reads its
// hash rows from the page cache and dequantizes them to BF16 for one device upload per call.
class PleRowReader;

class PleGather {
public:
    // `stream` reads rows with batched direct I/O instead of faulting them through the mapping,
    // so the table never occupies the page cache. `cache_bytes` then retains that many bytes of
    // already-read pages between gathers, so repeated n-grams stop re-reading the volume; 0
    // disables retention.
    PleGather(const PleTable& table, const PleConfig& config, bool stream,
              std::size_t cache_bytes = 0);
    ~PleGather();

    [[nodiscard]] bool streamed() const noexcept { return reader_ != nullptr; }

    [[nodiscard]] std::uint32_t width() const noexcept { return width_; }

    // BF16 `[width, count]` embedding columns of `tokens[first, first + count)`. Tokens before
    // `first` (and EOS before the sequence start) form each n-gram context, and an EOS closes
    // every window it falls into. `out` holds `width * count` BF16 words.
    void gather(std::span<const std::int32_t> tokens, std::size_t first, std::size_t count,
                std::span<std::uint16_t> out);

    // Starts filling the retained-page cache for the rows `gather(tokens, first, count)` would
    // read, on the reader's worker pool, without producing embedding columns; a following gather
    // of the same window then finds its pages resident and only dequantizes. Row indices and
    // ranges are computed synchronously on the calling thread (cheap); only the page reads run
    // asynchronously. Advisory only — a gather never depends on it. No-op without streaming and
    // retention.
    void prefetch(std::span<const std::int32_t> tokens, std::size_t first, std::size_t count);

    [[nodiscard]] bool retains() const noexcept;

    // Table rows of one position whose context tokens at distance 0..ngram_size-1 are `context`.
    void rows(std::span<const std::int64_t> context, std::span<std::uint64_t> out) const;

private:
    void column_rows(std::span<const std::int32_t> tokens, std::size_t position,
                     std::span<std::uint64_t> rows) const;
    void row_ranges(std::span<const std::uint64_t> rows, std::vector<PleRowReader::Range>& out) const;
    void decode_row(const std::byte* codes, const std::byte* scale, std::uint16_t* out) const;
    [[nodiscard]] const std::byte* locate(std::uint64_t object_offset, std::uint64_t bytes) const;

    PleTable table_;
    PleConfig config_;
    std::uint32_t width_     = 0;
    std::uint32_t row_width_ = 0;
    std::array<float, 256> e4m3_{};
    std::unique_ptr<PleRowReader> reader_;
};

} // namespace ninfer::models::qwen3_5::execution
