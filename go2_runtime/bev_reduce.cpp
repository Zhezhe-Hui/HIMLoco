// Exact per-cell reductions of all valid projected points; no decimation.
#include <algorithm>
#include <cstddef>
#include <cstdint>

extern "C" int go2_bev_reduce(std::size_t n, std::size_t cells,
    const std::int64_t* ids, const double* heights, float* lo, float* hi,
    std::int32_t* count, std::int32_t* elevated) {
    for (std::size_t i = 0; i < n; ++i) {
        if (ids[i] < 0 || static_cast<std::size_t>(ids[i]) >= cells) return 1;
        const auto cell = ids[i];
        const float h = static_cast<float>(heights[i]);
        lo[cell] = std::min(lo[cell], h);
        hi[cell] = std::max(hi[cell], h);
        ++count[cell];
        if (heights[i] > 0.15 && heights[i] < 1.20) ++elevated[cell];
    }
    return 0;
}
