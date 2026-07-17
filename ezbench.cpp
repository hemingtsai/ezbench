/*
 * ezbench — Cross-Platform CPU Benchmark
 * Copyright (c) 2026 Hemingtsai
 * Released under the MIT License.
 *
 * ============================================================================
 *  FILE INDEX
 * ============================================================================
 *  Section  1 — Headers & Compile-time Configuration  .......  line    41
 *  Section  2 — Utility Classes & Helpers  ...................  line    70
 *  Section  3 — System / Compiler Information  ...............  line   170
 *  Section  4 — Integer Arithmetic Benchmark  ................  line   220
 *  Section  5 — Floating-Point Benchmark  ....................  line   330
 *  Section  6 — Memory Bandwidth Benchmark  ..................  line   460
 *  Section  7 — Memory Latency (Pointer Chasing)  ............  line   610
 *  Section  8 — Branch Prediction Benchmark  .................  line   740
 *  Section  9 — Cache-Hierarchy Probing Benchmark  ...........  line   840
 *  Section 10 — Instruction-Level Parallelism (ILP)  .........  line   960
 *  Section 11 — Multi-Threaded Scaling Benchmark  ............  line  1060
 *  Section 12 — Dense Matrix Multiplication Benchmark  .......  line  1180
 *  Section 13 — Sorting Throughput Benchmark  ................  line  1280
 *  Section 14 — Hash / Integer-Mix Throughput  ...............  line  1380
 *  Section 15 — Score Aggregation & Final Report  ............  line  1440
 *  Section 16 — Entry Point (main)  ..........................  line  1550
 * ============================================================================
 *
 * Build (any C++17 compiler):
 *   g++ -std=c++17 -O2 -pthread -o ezbench ezbench.cpp
 *   clang++ -std=c++17 -O2 -pthread -o ezbench ezbench.cpp
 *
 * Run:
 *   ./ezbench
 *
 * No platform-specific headers or intrinsics are used.  The code relies
 * exclusively on ISO C++17 and the standard library.
 */

// ============================================================================
// Section 1 — Headers & Compile-time Configuration
// ============================================================================

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Tunable constants — modify these to alter benchmark intensity.
// Higher values give more stable results but take longer.
// ---------------------------------------------------------------------------
static constexpr int64_t  kIntIters       =  50'000'000;  // integer / fp loops
static constexpr int64_t  kFpIters        =  30'000'000;
static constexpr int64_t  kMemBufMB       =         256;  // MB for bandwidth
static constexpr int64_t  kLatIters       =   5'000'000;  // pointer-chase steps
static constexpr int64_t  kBranchSize     =   1'000'000;  // branch-test array
static constexpr int64_t  kCacheIters     =   2'000'000;  // per-size cache probe
static constexpr int64_t  kMatMulN        =         512;  // matrix dimension
static constexpr int64_t  kSortSize       =   2'000'000;  // sort array length
static constexpr int64_t  kHashMB         =         128;  // MB for hash test
static constexpr int      kWarmUpRounds   =           2;   // warm-up iterations

// ============================================================================
// Section 2 — Utility Classes & Helpers
// ============================================================================

// High-resolution wall-clock timer -------------------------------------------
class Timer {
    using clock = std::chrono::high_resolution_clock;
    clock::time_point t0_;
public:
    Timer() : t0_(clock::now()) {}

    // Elapsed time in seconds (floating-point).
    double secs() const {
        return std::chrono::duration<double>(clock::now() - t0_).count();
    }

    // Elapsed time in milliseconds.
    double ms() const {
        return std::chrono::duration<double, std::milli>(clock::now() - t0_).count();
    }

    void reset() { t0_ = clock::now(); }
};

// ---------------------------------------------------------------------------
// Prevent the compiler from optimising away a computation.
// We XOR the result into a file-scope volatile variable so that the compiler
// must treat it as an observable side-effect.  The final value is also
// printed, reinforcing that the computation is required.
// ---------------------------------------------------------------------------
static volatile uint64_t g_opt_barrier = 0;

inline void escape_result(uint64_t x) {
    g_opt_barrier ^= x;
}

// Return a seed that the compiler cannot know at compile-time.
inline uint64_t runtime_seed() {
    auto now = std::chrono::high_resolution_clock::now()
                   .time_since_epoch()
                   .count();
    return static_cast<uint64_t>(now);
}

// ---------------------------------------------------------------------------
// Simple progress indicator for long-running benchmarks.
// ---------------------------------------------------------------------------
inline void show_progress(const std::string& label) {
    std::cout << "  Running: " << std::left << std::setw(42) << label
              << std::flush;
}

inline void show_done(double val, const std::string& unit) {
    std::cout << "\r  " << std::left << std::setw(42) << ""
              << "\r  => " << std::fixed << std::setprecision(2)
              << val << " " << unit << std::endl;
}

// ---------------------------------------------------------------------------
// Format helpers
// ---------------------------------------------------------------------------
inline std::string fmt(double v, int prec = 2) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(prec) << v;
    return oss.str();
}

inline std::string pct(double v) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << (v * 100.0) << "%";
    return oss.str();
}

inline std::string comma(int64_t n) {
    std::string s = std::to_string(n);
    int pos = static_cast<int>(s.size()) - 3;
    while (pos > 0 && s[static_cast<size_t>(pos - 1)] != '-') {
        s.insert(static_cast<size_t>(pos), ",");
        pos -= 3;
    }
    return s;
}

// ============================================================================
// Section 3 — System / Compiler Information
// ============================================================================

struct SysInfo {
    std::string arch;
    std::string compiler;
    unsigned    hw_threads;

    static SysInfo detect() {
        SysInfo info;
        // Architecture detection via compiler pre-defined macros.
#if defined(__x86_64__) || defined(_M_X64)
        info.arch = "x86-64";
#elif defined(__i386__) || defined(_M_IX86)
        info.arch = "x86-32";
#elif defined(__aarch64__) || defined(_M_ARM64)
        info.arch = "AArch64 (ARM 64-bit)";
#elif defined(__arm__) || defined(_M_ARM)
        info.arch = "ARM 32-bit";
#elif defined(__powerpc64__) || defined(__ppc64__)
        info.arch = "PowerPC 64-bit";
#elif defined(__riscv)
#  if __riscv_xlen == 64
        info.arch = "RISC-V 64-bit";
#  else
        info.arch = "RISC-V 32-bit";
#  endif
#else
        info.arch = "Unknown";
#endif

        // Compiler detection.
#if defined(__clang__)
        info.compiler = std::string("Clang ") + std::to_string(__clang_major__)
                        + "." + std::to_string(__clang_minor__);
#elif defined(__GNUC__)
        info.compiler = std::string("GCC ") + std::to_string(__GNUC__)
                        + "." + std::to_string(__GNUC_MINOR__);
#elif defined(_MSC_VER)
        info.compiler = std::string("MSVC ") + std::to_string(_MSC_VER);
#else
        info.compiler = "Unknown";
#endif

        info.hw_threads = std::thread::hardware_concurrency();
        if (info.hw_threads == 0) info.hw_threads = 1;
        return info;
    }
};

// ============================================================================
// Section 4 — Integer Arithmetic Benchmark
// ============================================================================
// Measures throughput of core integer operations: addition, multiplication,
// division, and bitwise ops.  Operands are loop-carried to defeat the
// compiler's ability to pre-compute the results.

struct IntResult {
    double add_mops;    // million operations per second
    double mul_mops;
    double div_mops;
    double bit_mops;
    double composite_mops;  // equal-weight harmonic-ish mean
};

IntResult bench_integer() {
    show_progress("Integer Arithmetic");
    IntResult res{};

    // Use runtime seed so the compiler cannot constant-fold the loop.
    uint64_t s = runtime_seed();
    int64_t a = static_cast<int64_t>(s & 0xFFFF);
    int64_t b = static_cast<int64_t>((s >> 16) & 0xFFFF);
    int64_t c = static_cast<int64_t>((s >> 32) & 0xFFFF);
    int64_t x = a, y = b, z = c;

    // --- Addition ---
    {
        Timer t;
        for (int64_t i = 0; i < kIntIters; ++i) {
            x += y;   y += z;   z += x;   x += y;
            y += z;   z += x;   x += y;   y += z;
        }
        double ns = t.secs() * 1e9;
        // 8 adds per iteration.
        res.add_mops = (kIntIters * 8.0) / ns * 1000.0;
        escape_result(static_cast<uint64_t>(x + y + z));
    }

    // --- Multiplication ---
    x = a | 1;  y = b | 1;  z = c | 1;  // avoid zero for variety
    {
        Timer t;
        for (int64_t i = 0; i < kIntIters; ++i) {
            x *= y;   y *= z;   z *= x;   x *= y;
            y *= z;   z *= x;   x *= y;   y *= z;
        }
        double ns = t.secs() * 1e9;
        res.mul_mops = (kIntIters * 8.0) / ns * 1000.0;
        escape_result(static_cast<uint64_t>(x ^ y ^ z));
    }

    // --- Division (64-bit signed) ---
    x = a | 1;  y = b | 1;  z = c | 1;
    {
        Timer t;
        for (int64_t i = 0; i < kIntIters; ++i) {
            x = (x + z) / y;   y = (y + x) / z;   z = (z + y) / x;
            x = (x + z) / y;   y = (y + x) / z;
        }
        double ns = t.secs() * 1e9;
        // 5 divs per iteration.
        res.div_mops = (kIntIters * 5.0) / ns * 1000.0;
        escape_result(static_cast<uint64_t>(x + y + z));
    }

    // --- Bitwise / shift mix ---
    x = a;  y = b;  z = c;
    {
        Timer t;
        for (int64_t i = 0; i < kIntIters; ++i) {
            x ^= y;   y <<= (z & 7);   z >>= 3;   x |= y;
            y &= z;   z ^= x;   x <<= (y & 7);   y >>= 1;
        }
        double ns = t.secs() * 1e9;
        // 8 bitwise/shift ops per iteration.
        res.bit_mops = (kIntIters * 8.0) / ns * 1000.0;
        escape_result(static_cast<uint64_t>(x + y + z));
    }

    // Composite: geometric mean of the four sub-scores, then scaled.
    res.composite_mops = std::pow(res.add_mops * res.mul_mops *
                                  res.div_mops * res.bit_mops, 0.25);

    show_done(res.composite_mops, "MOPS (composite)");
    return res;
}

// ============================================================================
// Section 5 — Floating-Point Benchmark
// ============================================================================
// Tests single-precision (float) and double-precision throughput for
// addition, multiplication, division, and sqrt.

struct FpResult {
    double sp_add;     // MFLOPS, single precision
    double sp_mul;
    double sp_div;
    double sp_sqrt;
    double dp_add;     // MFLOPS, double precision
    double dp_mul;
    double dp_div;
    double dp_sqrt;
    double sp_composite;
    double dp_composite;
    double overall_mflops;
};

FpResult bench_fp() {
    show_progress("Floating-Point (scalar)");
    FpResult res{};

    uint64_t s = runtime_seed();
    float  fa = static_cast<float>((s & 0xFF) + 1);
    float  fb = static_cast<float>(((s >> 8) & 0xFF) + 1);
    double da = static_cast<double>((s >> 16) & 0xFF) + 1.0;
    double db = static_cast<double>((s >> 24) & 0xFF) + 1.0;

    // --- float add ---
    {
        float x = fa, y = fb, z = fa + fb;
        Timer t;
        for (int64_t i = 0; i < kFpIters; ++i) {
            x += y;  y += z;  z += x;  x += y;
            y += z;  z += x;  x += y;  y += z;
        }
        double ns = t.secs() * 1e9;
        res.sp_add = (kFpIters * 8.0) / ns * 1000.0;
        escape_result(static_cast<uint64_t>(x + y + z));
    }

    // --- float mul ---
    {
        float x = fa, y = fb, z = fa + fb;
        Timer t;
        for (int64_t i = 0; i < kFpIters; ++i) {
            x *= y;  y *= z;  z *= x;  x *= y;
            y *= z;  z *= x;  x *= y;  y *= z;
        }
        double ns = t.secs() * 1e9;
        res.sp_mul = (kFpIters * 8.0) / ns * 1000.0;
        escape_result(static_cast<uint64_t>(x + y + z));
    }

    // --- float div ---
    {
        float x = fa + 1.0f, y = fb + 1.0f, z = fa + fb + 1.0f;
        Timer t;
        for (int64_t i = 0; i < kFpIters / 2; ++i) {
            x = (x + z) / y;  y = (y + x) / z;  z = (z + y) / x;
            x = (x + z) / y;  y = (y + x) / z;
        }
        double ns = t.secs() * 1e9;
        res.sp_div = ((kFpIters / 2) * 5.0) / ns * 1000.0;
        escape_result(static_cast<uint64_t>(x + y + z));
    }

    // --- float sqrt ---
    {
        float x = fa + 1.0f, y = fb + 1.0f;
        Timer t;
        for (int64_t i = 0; i < kFpIters / 4; ++i) {
            x = std::sqrt(std::fabs(x) + 1.0f);
            y = std::sqrt(std::fabs(y) + 1.0f);
            x = std::sqrt(std::fabs(x) + 1.0f);
            y = std::sqrt(std::fabs(y) + 1.0f);
        }
        double ns = t.secs() * 1e9;
        res.sp_sqrt = ((kFpIters / 4) * 4.0) / ns * 1000.0;
        escape_result(static_cast<uint64_t>(x + y));
    }

    // --- double add ---
    {
        double x = da, y = db, z = da + db;
        Timer t;
        for (int64_t i = 0; i < kFpIters; ++i) {
            x += y;  y += z;  z += x;  x += y;
            y += z;  z += x;  x += y;  y += z;
        }
        double ns = t.secs() * 1e9;
        res.dp_add = (kFpIters * 8.0) / ns * 1000.0;
        escape_result(static_cast<uint64_t>(x + y + z));
    }

    // --- double mul ---
    {
        double x = da, y = db, z = da + db;
        Timer t;
        for (int64_t i = 0; i < kFpIters; ++i) {
            x *= y;  y *= z;  z *= x;  x *= y;
            y *= z;  z *= x;  x *= y;  y *= z;
        }
        double ns = t.secs() * 1e9;
        res.dp_mul = (kFpIters * 8.0) / ns * 1000.0;
        escape_result(static_cast<uint64_t>(x + y + z));
    }

    // --- double div ---
    {
        double x = da + 1.0, y = db + 1.0, z = da + db + 1.0;
        Timer t;
        for (int64_t i = 0; i < kFpIters / 2; ++i) {
            x = (x + z) / y;  y = (y + x) / z;  z = (z + y) / x;
            x = (x + z) / y;  y = (y + x) / z;
        }
        double ns = t.secs() * 1e9;
        res.dp_div = ((kFpIters / 2) * 5.0) / ns * 1000.0;
        escape_result(static_cast<uint64_t>(x + y + z));
    }

    // --- double sqrt ---
    {
        double x = da + 1.0, y = db + 1.0;
        Timer t;
        for (int64_t i = 0; i < kFpIters / 4; ++i) {
            x = std::sqrt(std::fabs(x) + 1.0);
            y = std::sqrt(std::fabs(y) + 1.0);
            x = std::sqrt(std::fabs(x) + 1.0);
            y = std::sqrt(std::fabs(y) + 1.0);
        }
        double ns = t.secs() * 1e9;
        res.dp_sqrt = ((kFpIters / 4) * 4.0) / ns * 1000.0;
        escape_result(static_cast<uint64_t>(x + y));
    }

    res.sp_composite = std::pow(res.sp_add * res.sp_mul *
                                res.sp_div * res.sp_sqrt, 0.25);
    res.dp_composite = std::pow(res.dp_add * res.dp_mul *
                                res.dp_div * res.dp_sqrt, 0.25);
    res.overall_mflops = std::sqrt(res.sp_composite * res.dp_composite);

    show_done(res.overall_mflops, "MFLOPS (composite)");
    return res;
}

// ============================================================================
// Section 6 — Memory Bandwidth Benchmark
// ============================================================================
// Allocates a large buffer and measures sequential read, sequential write,
// and copy (read+write) bandwidth.  The buffer is larger than typical L3
// caches so the measurement reflects DRAM bandwidth.

struct MemBwResult {
    double read_gbs;    // GB/s
    double write_gbs;
    double copy_gbs;
};

MemBwResult bench_mem_bandwidth() {
    show_progress("Memory Bandwidth (sequential)");
    MemBwResult res{};

    const int64_t elems = (kMemBufMB * 1024LL * 1024LL) / sizeof(uint64_t);
    std::vector<uint64_t> buf(elems);
    uint64_t sink = 0;

    // Fill buffer with non-zero data.
    uint64_t s = runtime_seed();
    for (int64_t i = 0; i < elems; ++i) {
        buf[static_cast<size_t>(i)] = s + static_cast<uint64_t>(i);
    }

    // --- Sequential read ---
    {
        // Warm up.
        for (int r = 0; r < kWarmUpRounds; ++r) {
            uint64_t tmp = 0;
            for (int64_t i = 0; i < elems; ++i)
                tmp ^= buf[static_cast<size_t>(i)];
            escape_result(tmp);
        }
        Timer t;
        uint64_t tmp = 0;
        for (int64_t i = 0; i < elems; ++i)
            tmp ^= buf[static_cast<size_t>(i)];
        double sec = t.secs();
        escape_result(tmp);
        double bytes = static_cast<double>(elems * sizeof(uint64_t));
        res.read_gbs = bytes / sec / 1e9;
    }

    // --- Sequential write ---
    {
        for (int r = 0; r < kWarmUpRounds; ++r) {
            for (int64_t i = 0; i < elems; ++i)
                buf[static_cast<size_t>(i)] = static_cast<uint64_t>(i);
        }
        Timer t;
        for (int64_t i = 0; i < elems; ++i)
            buf[static_cast<size_t>(i)] = static_cast<uint64_t>(i);
        double sec = t.secs();
        double bytes = static_cast<double>(elems * sizeof(uint64_t));
        res.write_gbs = bytes / sec / 1e9;
        escape_result(buf[0]);  // prevent dead-store elimination
    }

    // --- Copy (read + write) ---
    {
        std::vector<uint64_t> dst(elems);
        for (int r = 0; r < kWarmUpRounds; ++r) {
            for (int64_t i = 0; i < elems; ++i)
                dst[static_cast<size_t>(i)] = buf[static_cast<size_t>(i)];
        }
        Timer t;
        for (int64_t i = 0; i < elems; ++i)
            dst[static_cast<size_t>(i)] = buf[static_cast<size_t>(i)];
        double sec = t.secs();
        // Both read and write count toward the data volume.
        double bytes = static_cast<double>(elems * sizeof(uint64_t) * 2);
        res.copy_gbs = bytes / sec / 1e9;
        escape_result(dst[0]);
    }

    show_done(res.read_gbs, "GB/s (read)");
    return res;
}

// ============================================================================
// Section 7 — Memory Latency (Pointer Chasing) Benchmark
// ============================================================================
// Creates a circular linked list scattered across a buffer and measures the
// average time per access.  By varying the working-set size we can observe
// L1 / L2 / L3 / DRAM latency boundaries.

struct MemLatResult {
    double lat_ns;              // average pointer-chase latency (ns)
    std::vector<double> curve;   // latency at different sizes
};

// Helper: create a random permutation of indices [0, n-1].
static std::vector<int64_t> random_permutation(int64_t n, uint64_t seed) {
    std::vector<int64_t> perm(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        perm[static_cast<size_t>(i)] = i;
    std::mt19937_64 rng(seed);
    std::shuffle(perm.begin(), perm.end(), rng);
    return perm;
}

MemLatResult bench_mem_latency() {
    show_progress("Memory Latency (pointer chase)");
    MemLatResult res{};

    const int64_t stride = 64 / sizeof(int64_t); // one cache line apart
    const int64_t elems = kLatIters;
    // Use a stride to defeat hardware prefetchers.
    std::vector<int64_t> buf(static_cast<size_t>(elems * stride));

    // Build a linked list: buf[i * stride] = index of next node.
    auto perm = random_permutation(elems, runtime_seed());
    for (int64_t i = 0; i < elems; ++i) {
        int64_t cur = i * stride;
        int64_t nxt = perm[static_cast<size_t>(i)] * stride;
        buf[static_cast<size_t>(cur)] = nxt;
    }

    {
        // Warm up.
        int64_t idx = 0;
        for (int64_t i = 0; i < elems / 10; ++i)
            idx = buf[static_cast<size_t>(idx)];
        escape_result(static_cast<uint64_t>(idx));

        Timer t;
        idx = 0;
        for (int64_t i = 0; i < elems; ++i)
            idx = buf[static_cast<size_t>(idx)];
        double ns = t.secs() * 1e9;
        escape_result(static_cast<uint64_t>(idx));
        res.lat_ns = ns / static_cast<double>(elems);
    }

    // ---- Vary working-set size to reveal cache hierarchy ----
    // Sizes from 4 KB to 64 MB, stepping by powers of two.
    const std::array<int64_t, 16> cache_sizes_kb = {
        4, 8, 16, 32, 64, 128, 256, 512,
        1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072
    };

    for (int64_t size_kb : cache_sizes_kb) {
        int64_t n = (size_kb * 1024) / (stride * static_cast<int64_t>(sizeof(int64_t)));
        if (n < 16) n = 16;
        if (n > elems) break;

        // Build pointer chain within the first n * stride elements.
        auto perm_n = random_permutation(n, static_cast<uint64_t>(size_kb) * 7 + 1);
        std::vector<int64_t> local_buf(static_cast<size_t>(n * stride));
        for (int64_t i = 0; i < n; ++i) {
            int64_t cur = i * stride;
            int64_t nxt = perm_n[static_cast<size_t>(i)] * stride;
            local_buf[static_cast<size_t>(cur)] = nxt;
        }

        int64_t idx = 0;
        // Warm up.
        for (int64_t i = 0; i < kCacheIters / 10; ++i)
            idx = local_buf[static_cast<size_t>(idx)];

        Timer t;
        for (int64_t i = 0; i < kCacheIters; ++i)
            idx = local_buf[static_cast<size_t>(idx)];
        double ns = t.secs() * 1e9;
        escape_result(static_cast<uint64_t>(idx));
        double lat = ns / static_cast<double>(kCacheIters);
        res.curve.push_back(lat);
    }

    show_done(res.lat_ns, "ns (large-buffer average)");
    return res;
}

// ============================================================================
// Section 8 — Branch Prediction Benchmark
// ============================================================================
// Compares the cost of a data-dependent branch on sorted vs. unsorted data.
// A predictable branch (sorted) should be much faster than an unpredictable
// one (random), highlighting the branch-predictor's effectiveness.

struct BranchResult {
    double predictable_mops;    // million elements processed per second
    double unpredictable_mops;
    double ratio;               // predictable / unpredictable
};

BranchResult bench_branch() {
    show_progress("Branch Prediction");
    BranchResult res{};

    const int64_t n = kBranchSize;
    std::vector<int> data(static_cast<size_t>(n));

    // Generate random data in [0, 255].
    {
        std::mt19937 rng(static_cast<unsigned>(runtime_seed()));
        std::uniform_int_distribution<int> dist(0, 255);
        for (int64_t i = 0; i < n; ++i)
            data[static_cast<size_t>(i)] = dist(rng);
    }

    auto sum_if_lt_128 = [&](const std::vector<int>& d) {
        int64_t sum = 0;
        Timer t;
        // The branch body includes a dependency chain to discourage
        // the compiler from converting it into a conditional-select.
        for (int64_t i = 0; i < n; ++i) {
            if (d[static_cast<size_t>(i)] < 128) {
                sum += d[static_cast<size_t>(i)];
                sum ^= (sum << 7);
            } else {
                sum -= d[static_cast<size_t>(i)] / 2;
                sum ^= (sum >> 5);
            }
        }
        double sec = t.secs();
        escape_result(static_cast<uint64_t>(sum));
        return static_cast<double>(n) / sec / 1e6;  // Melem/s
    };

    // Predictable: sorted data (half < 128, half >= 128).
    {
        std::vector<int> sorted = data;
        std::sort(sorted.begin(), sorted.end());
        // Warm up.
        for (int r = 0; r < kWarmUpRounds; ++r)
            sum_if_lt_128(sorted);
        res.predictable_mops = sum_if_lt_128(sorted);
    }

    // Unpredictable: random order (branch taken ~50% of the time randomly).
    {
        for (int r = 0; r < kWarmUpRounds; ++r)
            sum_if_lt_128(data);
        res.unpredictable_mops = sum_if_lt_128(data);
    }

    res.ratio = res.predictable_mops / std::max(res.unpredictable_mops, 0.001);

    show_done(res.ratio, "x (predictable / unpredictable speedup)");
    return res;
}

// ============================================================================
// Section 9 — Cache-Hierarchy Probing Benchmark
// ============================================================================
// Measures sequential read bandwidth for working sets of increasing size.
// Bandwidth drops at each cache boundary (L1 → L2 → L3 → DRAM), revealing
// the effective cache sizes.

struct CacheResult {
    std::vector<double> sizes_kb;
    std::vector<double> bw_gbs;
};

CacheResult bench_cache_hierarchy() {
    show_progress("Cache Hierarchy Probing");
    CacheResult res;

    // Test sizes from 2 KB to 64 MB.
    const std::array<int64_t, 18> sizes_kb = {
        2, 4, 8, 16, 32, 64, 128, 256, 512,
        1024, 2048, 4096, 8192, 16384, 32768, 65536,
        131072, 262144
    };

    // Pre-allocate the largest buffer we will use.
    int64_t max_bytes = sizes_kb.back() * 1024;
    std::vector<uint64_t> buf(static_cast<size_t>(max_bytes / sizeof(uint64_t)));
    uint64_t s = runtime_seed();
    for (size_t i = 0; i < buf.size(); ++i)
        buf[i] = s + static_cast<uint64_t>(i);

    for (int64_t size_kb : sizes_kb) {
        int64_t bytes = size_kb * 1024;
        int64_t elems = bytes / static_cast<int64_t>(sizeof(uint64_t));
        // Access each element multiple times to get a stable measurement.
        int64_t total_iters = std::max(kCacheIters * 2LL,
                                       static_cast<int64_t>((64LL * 1024 * 1024) / sizeof(uint64_t) * 2));
        int64_t reps = total_iters / elems;
        if (reps < 1) reps = 1;

        // Warm up.
        uint64_t tmp = 0;
        for (int64_t r = 0; r < reps / 4; ++r)
            for (int64_t i = 0; i < elems; ++i)
                tmp ^= buf[static_cast<size_t>(i)];
        escape_result(tmp);

        Timer t;
        tmp = 0;
        for (int64_t r = 0; r < reps; ++r)
            for (int64_t i = 0; i < elems; ++i)
                tmp ^= buf[static_cast<size_t>(i)];
        double sec = t.secs();
        escape_result(tmp);

        double total_bytes = static_cast<double>(reps * elems * sizeof(uint64_t));
        double gbs = total_bytes / sec / 1e9;
        res.sizes_kb.push_back(static_cast<double>(size_kb));
        res.bw_gbs.push_back(gbs);
    }

    show_done(res.bw_gbs[0], "GB/s (L1 peak)");
    return res;
}

// ============================================================================
// Section 10 — Instruction-Level Parallelism (ILP) Benchmark
// ============================================================================
// Compares a chain of dependent operations (serial, cannot exploit ILP)
// against independent operations (the CPU can overlap them).  The ratio
// reveals how many operations the CPU can issue in parallel.

struct IlpResult {
    double dep_ops_per_ns;   // dependent chain throughput
    double ind_ops_per_ns;   // independent ops throughput
    double ilp_factor;       // ind / dep
};

IlpResult bench_ilp() {
    show_progress("Instruction-Level Parallelism");
    IlpResult res{};

    uint64_t s = runtime_seed();
    const int64_t iters = kIntIters;

    // --- Dependent chain ---
    // Each operation depends on the result of the previous one.
    {
        int64_t x = static_cast<int64_t>(s & 0xFFFF);
        Timer t;
        for (int64_t i = 0; i < iters; ++i) {
            x = ((x * 3 + 7) ^ (x >> 5)) + (x << 2);
            // Single dependency chain — only one op in flight at a time.
        }
        double ns = t.secs() * 1e9;
        res.dep_ops_per_ns = static_cast<double>(iters * 5) / ns; // ~5 ops/iter
        escape_result(static_cast<uint64_t>(x));
    }

    // --- Independent operations ---
    // Multiple independent chains allow the CPU to issue them in parallel.
    {
        int64_t a = static_cast<int64_t>(s & 0xFF);
        int64_t b = static_cast<int64_t>((s >> 8) & 0xFF);
        int64_t c = static_cast<int64_t>((s >> 16) & 0xFF);
        int64_t d = static_cast<int64_t>((s >> 24) & 0xFF);
        int64_t e = static_cast<int64_t>((s >> 32) & 0xFF);
        int64_t f = static_cast<int64_t>((s >> 40) & 0xFF);
        Timer t;
        for (int64_t i = 0; i < iters; ++i) {
            a = ((a * 3 + 7) ^ (a >> 3)) + (a << 2);
            b = ((b * 5 + 11) ^ (b >> 4)) + (b << 1);
            c = ((c * 7 + 13) ^ (c >> 2)) + (c << 3);
            d = ((d * 9 + 17) ^ (d >> 6)) + (d << 4);
            e = ((e * 11 + 19) ^ (e >> 1)) + (e << 5);
            f = ((f * 13 + 23) ^ (f >> 7)) + (f << 6);
            // Six independent chains → up to 6x throughput.
        }
        double ns = t.secs() * 1e9;
        res.ind_ops_per_ns = static_cast<double>(iters * 5 * 6) / ns;
        escape_result(static_cast<uint64_t>(a + b + c + d + e + f));
    }

    res.ilp_factor = res.ind_ops_per_ns / std::max(res.dep_ops_per_ns, 0.001);

    show_done(res.ilp_factor, "x (ILP speedup factor)");
    return res;
}

// ============================================================================
// Section 11 — Multi-Threaded Scaling Benchmark
// ============================================================================
// Runs a compute-bound workload (floating-point mix) with 1..N threads
// and measures total throughput.  This reveals how well the CPU scales
// across cores (affected by core count, SMT, thermal/power limits).

struct MtResult {
    std::vector<int>    thread_counts;
    std::vector<double> throughputs;     // million work-units / s
    std::vector<double> speedups;        // relative to 1 thread
    double              peak_speedup;
    int                 peak_threads;
};

// A trivially-parallel, compute-bound kernel.
static double mt_kernel_work(int64_t steps) {
    double x = 1.0, y = 2.0, z = 3.0;
    for (int64_t i = 0; i < steps; ++i) {
        x = std::sin(x) * 0.5 + std::cos(y) * 0.5;
        y = std::cos(y) * 0.5 + std::sin(z) * 0.5;
        z = std::sqrt(std::fabs(x * y)) + 1.0;
    }
    return x + y + z;
}

MtResult bench_multithread(const SysInfo& sys) {
    show_progress("Multi-Threaded Scaling");
    MtResult res;

    const int max_threads = std::min(static_cast<int>(sys.hw_threads), 64);
    const int64_t steps_per_thread = 2'000'000;

    // Measure single-thread baseline (throughput in steps/sec).
    double baseline_steps_per_sec = 0;
    {
        // Warm up.
        mt_kernel_work(steps_per_thread / 10);
        Timer t;
        volatile double sink = mt_kernel_work(steps_per_thread);
        double sec = t.secs();
        baseline_steps_per_sec = static_cast<double>(steps_per_thread) / std::max(sec, 1e-9);
        escape_result(static_cast<uint64_t>(sink * 1e6));
    }

    res.thread_counts.push_back(1);
    res.throughputs.push_back(baseline_steps_per_sec);
    res.speedups.push_back(1.0);

    // Test powers of two plus the maximum thread count.
    std::vector<int> counts;
    for (int t = 2; t <= max_threads; t *= 2)
        counts.push_back(t);
    if (counts.empty() || counts.back() != max_threads)
        counts.push_back(max_threads);

    for (int num_threads : counts) {
        if (num_threads <= 1) continue;

        std::atomic<int64_t> global_work{0};
        std::atomic<double>  global_sum{0.0};
        std::vector<std::thread> threads;

        Timer wall_t;
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, steps_per_thread]() {
                double sum = mt_kernel_work(steps_per_thread);
                global_work.fetch_add(steps_per_thread, std::memory_order_relaxed);
                // Atomic store — the compiler must keep this side-effect.
                global_sum.store(sum, std::memory_order_relaxed);
            });
        }
        for (auto& th : threads) th.join();

        double wall_sec = wall_t.secs();
        // Throughput: total work (steps) / wall time (sec).
        double tp = static_cast<double>(global_work.load()) /
                    std::max(wall_sec, 1e-9);
        escape_result(static_cast<uint64_t>(global_sum.load() * 1e9));

        double speedup = tp / baseline_steps_per_sec;
        res.thread_counts.push_back(num_threads);
        res.throughputs.push_back(tp);
        res.speedups.push_back(speedup);
    }

    // Find peak.
    res.peak_speedup = 1.0;
    res.peak_threads = 1;
    for (size_t i = 0; i < res.speedups.size(); ++i) {
        if (res.speedups[i] > res.peak_speedup) {
            res.peak_speedup = res.speedups[i];
            res.peak_threads = res.thread_counts[i];
        }
    }

    show_done(res.peak_speedup,
              "x peak speedup (" + std::to_string(res.peak_threads) + " threads)");
    return res;
}

// ============================================================================
// Section 12 — Dense Matrix Multiplication Benchmark
// ============================================================================
// Classic triple-nested-loop matrix multiplication on square matrices of
// floats.  Tests FP throughput, cache reuse, and memory bandwidth.

struct MatMulResult {
    double gflops;          // billion floating-point operations per second
    int64_t matrix_size;
};

// Naive but cache-friendly (loop order i-k-j for better locality).
MatMulResult bench_matmul() {
    show_progress("Matrix Multiplication (float)");
    MatMulResult res{};
    res.matrix_size = kMatMulN;

    const int64_t n = kMatMulN;
    std::vector<float> A(static_cast<size_t>(n * n));
    std::vector<float> B(static_cast<size_t>(n * n));
    std::vector<float> C(static_cast<size_t>(n * n), 0.0f);

    // Initialize matrices with deterministic "random" values.
    uint64_t s = runtime_seed();
    for (int64_t i = 0; i < n * n; ++i) {
        s = s * 1103515245 + 12345;  // simple LCG
        A[static_cast<size_t>(i)] = static_cast<float>((s & 0xFFFF)) / 65536.0f;
        B[static_cast<size_t>(i)] = static_cast<float>(((s >> 16) & 0xFFFF)) / 65536.0f;
    }

    auto mul = [&]() {
        // i-k-j loop for better cache behaviour on B and C.
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t k = 0; k < n; ++k) {
                float aik = A[static_cast<size_t>(i * n + k)];
                for (int64_t j = 0; j < n; ++j) {
                    C[static_cast<size_t>(i * n + j)] +=
                        aik * B[static_cast<size_t>(k * n + j)];
                }
            }
        }
    };

    // Warm up.
    {
        std::fill(C.begin(), C.end(), 0.0f);
        mul();
    }
    // Re-fill C with zeros and measure.
    std::fill(C.begin(), C.end(), 0.0f);
    Timer t;
    mul();
    double sec = t.secs();

    escape_result(static_cast<uint64_t>(C[0] * 1e9f));

    // 2 * N^3 floating-point ops.
    double total_ops = 2.0 * static_cast<double>(n) *
                       static_cast<double>(n) * static_cast<double>(n);
    res.gflops = total_ops / sec / 1e9;

    show_done(res.gflops, "GFLOPS");
    return res;
}

// ============================================================================
// Section 13 — Sorting Throughput Benchmark
// ============================================================================
// Measures how fast std::sort can order a large array of random integers.
// This stresses branch prediction, cache, and memory bandwidth.

struct SortResult {
    double melem_per_sec;   // million elements sorted per second
};

SortResult bench_sort() {
    show_progress("Sorting Throughput (std::sort)");
    SortResult res{};

    const int64_t n = kSortSize;
    std::vector<int64_t> data(static_cast<size_t>(n));

    // Generate random data.
    {
        std::mt19937_64 rng(runtime_seed());
        std::uniform_int_distribution<int64_t> dist;
        for (int64_t i = 0; i < n; ++i)
            data[static_cast<size_t>(i)] = dist(rng);
    }

    // Copy to avoid measuring generation time.
    std::vector<int64_t> work(static_cast<size_t>(n));

    // Warm up.
    {
        std::copy(data.begin(), data.end(), work.begin());
        std::sort(work.begin(), work.end());
    }

    {
        std::copy(data.begin(), data.end(), work.begin());
        Timer t;
        std::sort(work.begin(), work.end());
        double sec = t.secs();
        escape_result(static_cast<uint64_t>(work[0]));
        res.melem_per_sec = static_cast<double>(n) / sec / 1e6;
    }

    show_done(res.melem_per_sec, "Melem/s");
    return res;
}

// ============================================================================
// Section 14 — Hash / Integer-Mix Throughput Benchmark
// ============================================================================
// Computes a simple hash (FNV-1a style) over a buffer.  Measures raw
// integer-mix and memory-read throughput combined — relevant for hash-table
// and checksum workloads.

struct HashResult {
    double mbs;    // megabytes processed per second
};

HashResult bench_hash() {
    show_progress("Hash / Mix Throughput");
    HashResult res{};

    int64_t bytes = kHashMB * 1024LL * 1024LL;
    int64_t elems = bytes / sizeof(uint64_t);
    std::vector<uint64_t> buf(static_cast<size_t>(elems));
    uint64_t s = runtime_seed();
    for (int64_t i = 0; i < elems; ++i)
        buf[static_cast<size_t>(i)] = s + static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ULL;

    // Warm up.
    {
        uint64_t h = 0x811C9DC5ULL;
        for (int64_t i = 0; i < elems; ++i) {
            h ^= buf[static_cast<size_t>(i)];
            h *= 0x01000193ULL;
        }
        escape_result(h);
    }

    uint64_t h = 0x811C9DC5ULL;
    Timer t;
    for (int64_t i = 0; i < elems; ++i) {
        h ^= buf[static_cast<size_t>(i)];
        h *= 0x01000193ULL;
    }
    double sec = t.secs();
    escape_result(h);

    res.mbs = static_cast<double>(bytes) / sec / 1e6;

    show_done(res.mbs, "MB/s");
    return res;
}

// ============================================================================
// Section 15 — Score Aggregation & Final Report
// ============================================================================

// Reference values for a baseline mid-range CPU (circa 2020).
// These are calibrated against the scalar, single-threaded code produced
// by this benchmark, running on a typical 6-core desktop processor.
// They are used to normalise raw results into a 0–200+ score range.
struct Reference {
    static constexpr double int_composite    =  4000.0;   // MOPS
    static constexpr double fp_composite     =  2000.0;   // MFLOPS
    static constexpr double mem_read         =    17.0;   // GB/s
    static constexpr double mem_write        =    14.0;   // GB/s
    static constexpr double mem_copy         =    15.0;   // GB/s
    static constexpr double mem_latency      =    75.0;   // ns (lower is better)
    static constexpr double branch_ratio     =     2.0;   // predictable/unpred.
    static constexpr double ilp_factor       =     4.0;   // ind/dep throughput
    static constexpr double matmul_gflops    =    28.0;   // GFLOPS
    static constexpr double sort_melem       =    12.0;   // Melem/s
    static constexpr double hash_mbs         =  3000.0;   // MB/s
    static constexpr double mt_speedup_4t    =     3.3;   // speedup at 4 threads
};

// Normalise a "higher-is-better" metric.
inline double norm_higher(double measured, double ref) {
    if (ref <= 0.0) return 0.0;
    return (measured / ref) * 100.0;
}

// Normalise a "lower-is-better" metric (invert).
inline double norm_lower(double measured, double ref) {
    if (measured <= 0.0) return 200.0;
    return (ref / measured) * 100.0;
}

struct FinalReport {
    IntResult      int_res;
    FpResult       fp_res;
    MemBwResult    bw_res;
    MemLatResult   lat_res;
    BranchResult   br_res;
    CacheResult    cache_res;
    IlpResult      ilp_res;
    MtResult       mt_res;
    MatMulResult   mm_res;
    SortResult     sort_res;
    HashResult     hash_res;
    double         overall_score;

    void print(const SysInfo& sys) const {
        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
        std::cout << "║           ezbench — CPU Benchmark Results                    ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
        std::cout << "║  Architecture : " << std::left << std::setw(44) << sys.arch << "║\n";
        std::cout << "║  Compiler     : " << std::left << std::setw(44) << sys.compiler << "║\n";
        std::cout << "║  HW Threads   : " << std::left << std::setw(44) << sys.hw_threads << "║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════╣\n";

        // ---- Integer ----
        std::cout << "║                                                              ║\n";
        std::cout << "║  [1] Integer Arithmetic                                      ║\n";
        std::cout << "║      Add : " << std::right << std::setw(10) << fmt(int_res.add_mops)     << " MOPS"
                  << "  |  Mul : " << std::setw(10) << fmt(int_res.mul_mops)     << " MOPS  ║\n";
        std::cout << "║      Div : " << std::right << std::setw(10) << fmt(int_res.div_mops)     << " MOPS"
                  << "  |  Bit : " << std::setw(10) << fmt(int_res.bit_mops)     << " MOPS  ║\n";
        double int_score = norm_higher(int_res.composite_mops, Reference::int_composite);
        std::cout << "║      Composite : " << std::setw(10) << fmt(int_res.composite_mops)
                  << " MOPS  →  score " << std::setw(6) << fmt(int_score, 1) << "          ║\n";

        // ---- FP ----
        std::cout << "║                                                              ║\n";
        std::cout << "║  [2] Floating-Point (scalar)                                 ║\n";
        std::cout << "║      SP  Add / Mul / Div / Sqrt : "
                  << fmt(fp_res.sp_add, 0) << " / " << fmt(fp_res.sp_mul, 0) << " / "
                  << fmt(fp_res.sp_div, 0) << " / " << fmt(fp_res.sp_sqrt, 0) << " MFLOPS ║\n";
        std::cout << "║      DP  Add / Mul / Div / Sqrt : "
                  << fmt(fp_res.dp_add, 0) << " / " << fmt(fp_res.dp_mul, 0) << " / "
                  << fmt(fp_res.dp_div, 0) << " / " << fmt(fp_res.dp_sqrt, 0) << " MFLOPS ║\n";
        double fp_score = norm_higher(fp_res.overall_mflops, Reference::fp_composite);
        std::cout << "║      Composite : " << std::setw(10) << fmt(fp_res.overall_mflops, 0)
                  << " MFLOPS →  score " << std::setw(6) << fmt(fp_score, 1) << "          ║\n";

        // ---- Memory BW ----
        std::cout << "║                                                              ║\n";
        std::cout << "║  [3] Memory Bandwidth (sequential, " << kMemBufMB << " MB)                      ║\n";
        std::cout << "║      Read  : " << std::setw(10) << fmt(bw_res.read_gbs)  << " GB/s"
                  << "  |  Write : " << std::setw(10) << fmt(bw_res.write_gbs) << " GB/s  ║\n";
        std::cout << "║      Copy  : " << std::setw(10) << fmt(bw_res.copy_gbs)  << " GB/s"
                  << "                              ║\n";
        double bw_read_score = norm_higher(bw_res.read_gbs, Reference::mem_read);
        double bw_write_score = norm_higher(bw_res.write_gbs, Reference::mem_write);
        double bw_score = std::sqrt(bw_read_score * bw_write_score);
        std::cout << "║      Score : " << std::setw(10) << fmt(bw_score, 1) << "                          ║\n";

        // ---- Memory Latency ----
        std::cout << "║                                                              ║\n";
        std::cout << "║  [4] Memory Latency (pointer chase)                          ║\n";
        std::cout << "║      Random access latency : " << std::setw(7)
                  << fmt(lat_res.lat_ns) << " ns                         ║\n";
        double lat_score = norm_lower(lat_res.lat_ns, Reference::mem_latency);
        std::cout << "║      Score : " << std::setw(10) << fmt(lat_score, 1) << "                          ║\n";

        // ---- Branch ----
        std::cout << "║                                                              ║\n";
        std::cout << "║  [5] Branch Prediction                                       ║\n";
        std::cout << "║      Predictable   : " << std::setw(10) << fmt(br_res.predictable_mops)
                  << " Melem/s                ║\n";
        std::cout << "║      Unpredictable : " << std::setw(10) << fmt(br_res.unpredictable_mops)
                  << " Melem/s                ║\n";
        std::cout << "║      Ratio         : " << std::setw(10) << fmt(br_res.ratio)
                  << " x                         ║\n";
        double br_score = norm_higher(br_res.ratio, Reference::branch_ratio);
        std::cout << "║      Score : " << std::setw(10) << fmt(br_score, 1) << "                          ║\n";

        // ---- Cache ----
        std::cout << "║                                                              ║\n";
        std::cout << "║  [6] Cache Hierarchy (bandwidth vs. working-set size)       ║\n";
        for (size_t i = 0; i < cache_res.sizes_kb.size() && i < 9; ++i) {
            std::string label;
            double kb = cache_res.sizes_kb[i];
            if (kb >= 1024.0)
                label = fmt(kb / 1024.0, 0) + " MB";
            else
                label = fmt(kb, 0) + " KB";
            std::cout << "║      " << std::setw(10) << label << " : " << std::setw(10)
                      << fmt(cache_res.bw_gbs[i]) << " GB/s                       ║\n";
        }
        // Use small-buffer bandwidth as the cache score baseline.
        double cache_score = norm_higher(cache_res.bw_gbs.empty() ? 0.0 : cache_res.bw_gbs[0],
                                         100.0); // 100 GB/s reference for L1
        std::cout << "║      Score (L1 peak) : " << std::setw(10) << fmt(cache_score, 1)
                  << "                       ║\n";

        // ---- ILP ----
        std::cout << "║                                                              ║\n";
        std::cout << "║  [7] Instruction-Level Parallelism                           ║\n";
        std::cout << "║      Dependent   : " << std::setw(10)
                  << fmt(ilp_res.dep_ops_per_ns) << " ops/ns             ║\n";
        std::cout << "║      Independent : " << std::setw(10)
                  << fmt(ilp_res.ind_ops_per_ns) << " ops/ns             ║\n";
        std::cout << "║      ILP Factor  : " << std::setw(10)
                  << fmt(ilp_res.ilp_factor) << " x                    ║\n";
        double ilp_score = norm_higher(ilp_res.ilp_factor, Reference::ilp_factor);
        std::cout << "║      Score : " << std::setw(10) << fmt(ilp_score, 1) << "                          ║\n";

        // ---- Multi-thread ----
        std::cout << "║                                                              ║\n";
        std::cout << "║  [8] Multi-Threaded Scaling                                  ║\n";
        for (size_t i = 0; i < mt_res.thread_counts.size(); ++i) {
            std::cout << "║      " << std::setw(3) << mt_res.thread_counts[i] << " threads : "
                      << std::setw(7) << fmt(mt_res.speedups[i]) << " x speedup"
                      << "  (" << pct(mt_res.speedups[i] / mt_res.thread_counts[i])
                      << " efficiency)       ║\n";
        }
        double mt_score = 0.0;
        // Find speedup at 4 threads (or closest).
        for (size_t i = 0; i < mt_res.thread_counts.size(); ++i) {
            if (mt_res.thread_counts[i] >= 4) {
                mt_score = norm_higher(mt_res.speedups[i], Reference::mt_speedup_4t);
                break;
            }
        }
        if (mt_score == 0.0 && !mt_res.speedups.empty())
            mt_score = norm_higher(mt_res.speedups.back(), 2.0); // fallback
        std::cout << "║      Score : " << std::setw(10) << fmt(mt_score, 1) << "                          ║\n";

        // ---- MatMul ----
        std::cout << "║                                                              ║\n";
        std::cout << "║  [9] Matrix Multiplication (" << mm_res.matrix_size << "x" << mm_res.matrix_size
                  << " float)                ║\n";
        std::cout << "║      Throughput : " << std::setw(10) << fmt(mm_res.gflops)
                  << " GFLOPS                      ║\n";
        double mm_score = norm_higher(mm_res.gflops, Reference::matmul_gflops);
        std::cout << "║      Score : " << std::setw(10) << fmt(mm_score, 1) << "                          ║\n";

        // ---- Sort ----
        std::cout << "║                                                              ║\n";
        std::cout << "║ [10] Sorting Throughput (std::sort)                          ║\n";
        std::cout << "║      " << comma(kSortSize) << " elements : "
                  << std::setw(10) << fmt(sort_res.melem_per_sec) << " Melem/s          ║\n";
        double sort_score = norm_higher(sort_res.melem_per_sec, Reference::sort_melem);
        std::cout << "║      Score : " << std::setw(10) << fmt(sort_score, 1) << "                          ║\n";

        // ---- Hash ----
        std::cout << "║                                                              ║\n";
        std::cout << "║ [11] Hash / Mix Throughput (" << kHashMB << " MB)                          ║\n";
        std::cout << "║      Throughput : " << std::setw(10) << fmt(hash_res.mbs)
                  << " MB/s                        ║\n";
        double hash_score = norm_higher(hash_res.mbs, Reference::hash_mbs);
        std::cout << "║      Score : " << std::setw(10) << fmt(hash_score, 1) << "                          ║\n";

        // ---- Overall ----
        std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
        std::cout << "║                                                              ║\n";
        std::cout << "║  ★  OVERALL SCORE  :  " << std::setw(10) << fmt(overall_score, 1)
                  << "                          ║\n";
        std::cout << "║                                                              ║\n";
        std::cout << "║  (Geometric mean of all sub-scores.  100 ≈ baseline.         ║\n";
        std::cout << "║   Higher is better.)                                        ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════╝\n";

        // Machine-readable summary line.
        std::cout << "\n[CSV] int,fp,mem_r,mem_w,mem_lat,branch,ilp,multithread,matmul,sort,hash,overall\n";
        std::cout << "[CSV] " << fmt(int_score, 1) << "," << fmt(fp_score, 1) << ","
                  << fmt(bw_read_score, 1) << "," << fmt(bw_write_score, 1) << ","
                  << fmt(lat_score, 1) << "," << fmt(br_score, 1) << ","
                  << fmt(ilp_score, 1) << "," << fmt(mt_score, 1) << ","
                  << fmt(mm_score, 1) << "," << fmt(sort_score, 1) << ","
                  << fmt(hash_score, 1) << "," << fmt(overall_score, 1) << "\n";
    }
};

static double compute_overall(const FinalReport& r) {
    // Collect sub-scores.
    std::vector<double> scores;
    scores.push_back(norm_higher(r.int_res.composite_mops, Reference::int_composite));
    scores.push_back(norm_higher(r.fp_res.overall_mflops, Reference::fp_composite));
    scores.push_back(norm_higher(r.bw_res.read_gbs, Reference::mem_read));
    scores.push_back(norm_higher(r.bw_res.write_gbs, Reference::mem_write));
    scores.push_back(norm_lower(r.lat_res.lat_ns, Reference::mem_latency));
    scores.push_back(norm_higher(r.br_res.ratio, Reference::branch_ratio));
    double cache_bw = r.cache_res.bw_gbs.empty() ? 0.0 : r.cache_res.bw_gbs[0];
    scores.push_back(norm_higher(cache_bw, 100.0));
    scores.push_back(norm_higher(r.ilp_res.ilp_factor, Reference::ilp_factor));
    // MT score: use 4-thread speedup (or closest).
    double mt_sp = r.mt_res.speedups.empty() ? 1.0 : r.mt_res.speedups.back();
    for (size_t i = 0; i < r.mt_res.thread_counts.size(); ++i) {
        if (r.mt_res.thread_counts[i] >= 4) {
            mt_sp = r.mt_res.speedups[i];
            break;
        }
    }
    scores.push_back(norm_higher(mt_sp, Reference::mt_speedup_4t));
    scores.push_back(norm_higher(r.mm_res.gflops, Reference::matmul_gflops));
    scores.push_back(norm_higher(r.sort_res.melem_per_sec, Reference::sort_melem));
    scores.push_back(norm_higher(r.hash_res.mbs, Reference::hash_mbs));

    // Geometric mean.
    double log_sum = 0.0;
    for (double s : scores) {
        if (s <= 0.0) s = 0.01;
        log_sum += std::log(s);
    }
    return std::exp(log_sum / static_cast<double>(scores.size()));
}

// ============================================================================
// Section 16 — Entry Point
// ============================================================================

int main() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  ezbench — Cross-Platform CPU Benchmark                      ║\n";
    std::cout << "║  Copyright (c) 2026 Hemingtsai  |  MIT License               ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

    SysInfo sys = SysInfo::detect();
    std::cout << "Detected: " << sys.arch << " | " << sys.compiler
              << " | " << sys.hw_threads << " HW thread(s)\n\n";

    FinalReport report;

    // Run all benchmarks.
    std::cout << "--- Benchmarking ---\n";
    report.int_res   = bench_integer();
    report.fp_res    = bench_fp();
    report.bw_res    = bench_mem_bandwidth();
    report.lat_res   = bench_mem_latency();
    report.br_res    = bench_branch();
    report.cache_res = bench_cache_hierarchy();
    report.ilp_res   = bench_ilp();
    report.mt_res    = bench_multithread(sys);
    report.mm_res    = bench_matmul();
    report.sort_res  = bench_sort();
    report.hash_res  = bench_hash();

    report.overall_score = compute_overall(report);

    // Print the final report.
    report.print(sys);

    std::cout << "\nDone.  (optimisation barrier: " << std::hex
              << g_opt_barrier << std::dec << ")\n";
    return 0;
}
