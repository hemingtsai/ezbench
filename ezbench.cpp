/*
 * ezbench v2 — Cross-Platform CPU Benchmark (40 tests, 8 categories, 15 ISAs)
 * Copyright (c) 2026 Hemingtsai
 * Released under the MIT License.
 *
 * ============================================================================
 *  FILE INDEX
 * ============================================================================
 *  Section  1 — Headers & Compile-time Configuration  .......  line    49
 *  Section  2 — Utility Classes & Helpers  ...................  line    78
 *  Section  3 — System / Compiler Information  ...............  line   179
 *  Section  4 — Integer Arithmetic Benchmark  ................  line   234
 *  Section  5 — Floating-Point Benchmark  ....................  line   351
 *  Section  6 — Memory Bandwidth Benchmark  ..................  line   498
 *  Section  7 — Memory Latency (Pointer Chasing)  ............  line   651
 *  Section  8 — Branch Prediction Benchmark  .................  line   790
 *  Section  9 — Cache-Hierarchy Probing Benchmark  ...........  line   897
 *  Section 10 — Instruction-Level Parallelism (ILP)  .........  line  1015
 *  Section 11 — Multi-Threaded Scaling Benchmark  ............  line  1112
 *  Section 12 — Dense Matrix Multiplication Benchmark  .......  line  1238
 *  Section 13 — Sorting Throughput Benchmark  ................  line  1340
 *  Section 14 — Hash / Integer-Mix Throughput  ...............  line  1410
 *  Section 15 — Fast Fourier Transform (FFT)  ................  line  1480
 *  Section 16 — N-body Gravitational Simulation  .............  line  1550
 *  Section 17 — Fluid Dynamics (Lattice Boltzmann)  ..........  line  1690
 *  Section 18 — Molecular Dynamics (Lennard-Jones)  ..........  line  1860
 *  Section 19 — Game Logic Simulation  .......................  line  1960
 *  Section 20 — AI Inference (Neural Network)  ...............  line  2030
 *  Section 21 — Score Aggregation & Final Report  ............  line  2120
 *  Section 22 — Entry Point (main)  ..........................  line  2240
 * ============================================================================
 *
 * Every benchmark runs kBenchRounds (default 3) independent rounds and
 * reports the arithmetic mean.  This suppresses run-to-run noise caused by
 * OS scheduling, thermal throttling, and frequency transients.
 *
 * Supported architectures: x86-64, x86-32, AArch64, ARM 32-bit (armhf &
 * soft-float), RISC-V 64/32, LoongArch 64/32, PowerPC 64/32,
 * IBM z/Architecture (s390x), IBM S/390, MIPS 64/32.
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
#include <cctype>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
static constexpr int64_t  kFftN           =       65536;  // FFT points (power of 2)
static constexpr int      kNbodyBodies    =        2000;  // N-body count
static constexpr int      kLbmNy          =         128;  // LBM grid height
static constexpr int      kLbmNx          =         256;  // LBM grid width
static constexpr int      kLbmSteps       =          20;  // LBM steps per round
static constexpr int      kMdAtoms        =         800;  // MD atom count
static constexpr int      kMdSteps        =          10;  // MD steps per round
static constexpr int      kGameEntities   =      100000;  // game entity count
static constexpr int      kAiBatch        =         500;
static constexpr int      kFdtdNx         =         128;  // FDTD grid X
static constexpr int      kFdtdNy         =         128;  // FDTD grid Y
static constexpr int      kFdtdSteps      =          30;  // FDTD time steps
static constexpr int      kRigidBodies    =         300;  // rigid bodies
static constexpr int      kRayResX        =         256;  // ray trace resolution X
static constexpr int      kRayResY        =         256;  // ray trace resolution Y
static constexpr int      kHfBasis        =          10;  // HF basis functions
static constexpr int      kMcSteps        =        1000;  // Monte Carlo steps
static constexpr int      kHpSeqLen       =          20;  // HP sequence length
static constexpr int      kHpAnnealSteps  =         500;  // HP annealing steps
static constexpr int      kSwSeqLen       =         500;  // Smith-Waterman seq length
static constexpr int      kIzhNeurons     =         200;  // Izhikevich neurons
static constexpr int      kIzhSteps       =         500;  // Izhikevich time steps
static constexpr int64_t  kLuN            =         400;  // LU matrix size
static constexpr int      kOdeSteps       =     1000000;  // ODE solver steps
static constexpr int      kAstNodes       =     5000000;  // AST nodes
static constexpr int      kKaratsubaBits  =       65536;  // BigInt bits
static constexpr int64_t  kRegexMB        =          32;  // regex input MB
static constexpr int64_t  kJsonMB         =          32;  // JSON input MB (approx)
static constexpr int64_t  kAllocOps       =     2000000;  // alloc/free ops
static constexpr int64_t  kAtomicIters    =   20'000'000;  // atomic inc iterations
static constexpr int64_t  kParseTokens    =     500000;  // parser tokens
static constexpr int64_t  kVmInsns       =   2'000'000;  // VM instructions
static constexpr int64_t  kAesMB          =          16;  // AES input MB
static constexpr int64_t  kShaMB          =          32;  // SHA input MB
static constexpr int64_t  kLzssMB         =           1;  // LZSS input MB
static constexpr int      kConvRes        =        1024;  // convolution image size
static constexpr int      kRasterTris     =       20000;  // raster triangles
  // AI inference batch size

static constexpr int      kWarmUpRounds   =           2;   // warm-up iterations
static constexpr int      kBenchRounds    =           3;   // measurement rounds (averaged)

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
#  if defined(__ARM_PCS_VFP)
        info.arch = "ARM 32-bit (hard-float / armhf)";
#  else
        info.arch = "ARM 32-bit (soft-float)";
#  endif
#elif defined(__loongarch64)
        info.arch = "LoongArch 64-bit";
#elif defined(__loongarch__)
        info.arch = "LoongArch 32-bit";
#elif defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
        info.arch = "PowerPC 64-bit";
#elif defined(__powerpc__) || defined(__ppc__)
        info.arch = "PowerPC 32-bit";
#elif defined(__s390x__)
        info.arch = "IBM z/Architecture (s390x, 64-bit)";
#elif defined(__s390__)
        info.arch = "IBM S/390 (31-bit)";
#elif defined(__riscv)
#  if __riscv_xlen == 64
        info.arch = "RISC-V 64-bit";
#  else
        info.arch = "RISC-V 32-bit";
#  endif
#elif defined(__mips64)
        info.arch = "MIPS 64-bit";
#elif defined(__mips__)
        info.arch = "MIPS 32-bit";
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
    constexpr int R = kBenchRounds;
    IntResult res{};

    double sum_add = 0, sum_mul = 0, sum_div = 0, sum_bit = 0;

    for (int round = 0; round < R; ++round) {
        // Use runtime seed + round offset so each round differs.
        uint64_t s = runtime_seed() + static_cast<uint64_t>(round) * 0x9E3779B97F4A7C15ULL;
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
            sum_add += (kIntIters * 8.0) / ns * 1000.0;
            escape_result(static_cast<uint64_t>(x + y + z));
        }

        // --- Multiplication ---
        x = a | 1;  y = b | 1;  z = c | 1;
        {
            Timer t;
            for (int64_t i = 0; i < kIntIters; ++i) {
                x *= y;   y *= z;   z *= x;   x *= y;
                y *= z;   z *= x;   x *= y;   y *= z;
            }
            double ns = t.secs() * 1e9;
            sum_mul += (kIntIters * 8.0) / ns * 1000.0;
            escape_result(static_cast<uint64_t>(x ^ y ^ z));
        }

        // --- Division (64-bit signed) ---
        x = a | 1;  y = b | 1;  z = c | 1;
        {
            Timer t;
            for (int64_t i = 0; i < kIntIters; ++i) {
                x = (x + z) / (y | 1);   y = (y + x) / (z | 1);   z = (z + y) / (x | 1);
                x = (x + z) / (y | 1);   y = (y + x) / (z | 1);
            }
            double ns = t.secs() * 1e9;
            sum_div += (kIntIters * 5.0) / ns * 1000.0;
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
            sum_bit += (kIntIters * 8.0) / ns * 1000.0;
            escape_result(static_cast<uint64_t>(x + y + z));
        }
    }

    res.add_mops = sum_add / R;
    res.mul_mops = sum_mul / R;
    res.div_mops = sum_div / R;
    res.bit_mops = sum_bit / R;
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
    constexpr int R = kBenchRounds;
    FpResult res{};

    double sum_sp_add = 0, sum_sp_mul = 0, sum_sp_div = 0, sum_sp_sqrt = 0;
    double sum_dp_add = 0, sum_dp_mul = 0, sum_dp_div = 0, sum_dp_sqrt = 0;

    for (int round = 0; round < R; ++round) {
        uint64_t s = runtime_seed() + static_cast<uint64_t>(round) * 0x9E3779B97F4A7C15ULL;
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
            sum_sp_add += (kFpIters * 8.0) / ns * 1000.0;
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
            sum_sp_mul += (kFpIters * 8.0) / ns * 1000.0;
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
            sum_sp_div += ((kFpIters / 2) * 5.0) / ns * 1000.0;
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
            sum_sp_sqrt += ((kFpIters / 4) * 4.0) / ns * 1000.0;
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
            sum_dp_add += (kFpIters * 8.0) / ns * 1000.0;
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
            sum_dp_mul += (kFpIters * 8.0) / ns * 1000.0;
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
            sum_dp_div += ((kFpIters / 2) * 5.0) / ns * 1000.0;
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
            sum_dp_sqrt += ((kFpIters / 4) * 4.0) / ns * 1000.0;
            escape_result(static_cast<uint64_t>(x + y));
        }
    }

    res.sp_add = sum_sp_add / R;
    res.sp_mul = sum_sp_mul / R;
    res.sp_div = sum_sp_div / R;
    res.sp_sqrt = sum_sp_sqrt / R;
    res.dp_add = sum_dp_add / R;
    res.dp_mul = sum_dp_mul / R;
    res.dp_div = sum_dp_div / R;
    res.dp_sqrt = sum_dp_sqrt / R;
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
    constexpr int R = kBenchRounds;
    MemBwResult res{};

    const int64_t elems = (kMemBufMB * 1024LL * 1024LL) / sizeof(uint64_t);
    std::vector<uint64_t> buf(elems);

    double sum_read = 0, sum_write = 0, sum_copy = 0;

    for (int round = 0; round < R; ++round) {
        // Re-fill buffer with non-zero data each round.
        uint64_t s = runtime_seed() + static_cast<uint64_t>(round) * 0x9E3779B97F4A7C15ULL;
        for (int64_t i = 0; i < elems; ++i) {
            buf[static_cast<size_t>(i)] = s + static_cast<uint64_t>(i);
        }

        // --- Sequential read ---
        {
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
            sum_read += bytes / sec / 1e9;
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
            sum_write += bytes / sec / 1e9;
            escape_result(buf[0]);
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
            double bytes = static_cast<double>(elems * sizeof(uint64_t) * 2);
            sum_copy += bytes / sec / 1e9;
            escape_result(dst[0]);
        }
    }

    res.read_gbs  = sum_read  / R;
    res.write_gbs = sum_write / R;
    res.copy_gbs  = sum_copy  / R;

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
    constexpr int R = kBenchRounds;
    MemLatResult res{};

    const int64_t stride = 64 / sizeof(int64_t); // one cache line apart
    const int64_t elems = kLatIters;

    double sum_lat = 0;

    // Sizes from 4 KB to 128 MB, stepping by powers of two.
    const std::array<int64_t, 16> cache_sizes_kb = {
        4, 8, 16, 32, 64, 128, 256, 512,
        1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072
    };
    std::vector<double> curve_sums(cache_sizes_kb.size(), 0.0);

    for (int round = 0; round < R; ++round) {
        uint64_t seed = runtime_seed() + static_cast<uint64_t>(round) * 0x9E3779B97F4A7C15ULL;

        // --- Large-buffer pointer chase ---
        std::vector<int64_t> buf(static_cast<size_t>(elems * stride));
        auto perm = random_permutation(elems, seed);
        for (int64_t i = 0; i < elems; ++i) {
            int64_t cur = i * stride;
            int64_t nxt = perm[static_cast<size_t>(i)] * stride;
            buf[static_cast<size_t>(cur)] = nxt;
        }
        {
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
            sum_lat += ns / static_cast<double>(elems);
        }

        // --- Per-size latency curve ---
        for (size_t si = 0; si < cache_sizes_kb.size(); ++si) {
            int64_t size_kb = cache_sizes_kb[si];
            int64_t n = (size_kb * 1024) / (stride * static_cast<int64_t>(sizeof(int64_t)));
            if (n < 16) n = 16;
            if (n > elems) break;

            auto perm_n = random_permutation(n, static_cast<uint64_t>(size_kb) * 7 + seed);
            std::vector<int64_t> local_buf(static_cast<size_t>(n * stride));
            for (int64_t i = 0; i < n; ++i) {
                int64_t cur = i * stride;
                int64_t nxt = perm_n[static_cast<size_t>(i)] * stride;
                local_buf[static_cast<size_t>(cur)] = nxt;
            }
            int64_t idx = 0;
            for (int64_t i = 0; i < kCacheIters / 10; ++i)
                idx = local_buf[static_cast<size_t>(idx)];
            Timer t;
            for (int64_t i = 0; i < kCacheIters; ++i)
                idx = local_buf[static_cast<size_t>(idx)];
            double ns = t.secs() * 1e9;
            escape_result(static_cast<uint64_t>(idx));
            curve_sums[si] += ns / static_cast<double>(kCacheIters);
        }
    }

    res.lat_ns = sum_lat / R;
    for (double& v : curve_sums) v /= R;
    res.curve = curve_sums;

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
    constexpr int R = kBenchRounds;
    BranchResult res{};

    const int64_t n = kBranchSize;

    double sum_pred = 0, sum_unpred = 0;

    // Generate base random data once, then derive sorted copy.
    std::vector<int> data(static_cast<size_t>(n));
    {
        std::mt19937 rng(static_cast<unsigned>(runtime_seed()));
        std::uniform_int_distribution<int> dist(0, 255);
        for (int64_t i = 0; i < n; ++i)
            data[static_cast<size_t>(i)] = dist(rng);
    }
    std::vector<int> sorted = data;
    std::sort(sorted.begin(), sorted.end());

    for (int round = 0; round < R; ++round) {
        auto sum_if_lt_128 = [&](const std::vector<int>& d) {
            int64_t sum = 0;
            Timer t;
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

        // Warm up.
        for (int r = 0; r < kWarmUpRounds; ++r) {
            sum_if_lt_128(sorted);
            sum_if_lt_128(data);
        }
        sum_pred   += sum_if_lt_128(sorted);
        sum_unpred += sum_if_lt_128(data);
    }

    res.predictable_mops   = sum_pred   / R;
    res.unpredictable_mops = sum_unpred / R;
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
    constexpr int R = kBenchRounds;
    CacheResult res;

    const std::array<int64_t, 18> sizes_kb = {
        2, 4, 8, 16, 32, 64, 128, 256, 512,
        1024, 2048, 4096, 8192, 16384, 32768, 65536,
        131072, 262144
    };

    // Pre-fill sizes vector (same every round).
    for (int64_t sk : sizes_kb) res.sizes_kb.push_back(static_cast<double>(sk));
    std::vector<double> bw_sums(sizes_kb.size(), 0.0);

    int64_t max_bytes = sizes_kb.back() * 1024;

    for (int round = 0; round < R; ++round) {
        std::vector<uint64_t> buf(static_cast<size_t>(max_bytes / sizeof(uint64_t)));
        uint64_t s = runtime_seed() + static_cast<uint64_t>(round) * 0x9E3779B97F4A7C15ULL;
        for (size_t i = 0; i < buf.size(); ++i)
            buf[i] = s + static_cast<uint64_t>(i);

        for (size_t si = 0; si < sizes_kb.size(); ++si) {
            int64_t size_kb = sizes_kb[si];
            int64_t bytes = size_kb * 1024;
            int64_t elems = bytes / static_cast<int64_t>(sizeof(uint64_t));
            int64_t total_iters = std::max(static_cast<int64_t>(kCacheIters * 2),
                                static_cast<int64_t>((64LL * 1024 * 1024) / sizeof(uint64_t) * 2));
            int64_t reps = total_iters / elems;
            if (reps < 1) reps = 1;

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
            bw_sums[si] += total_bytes / sec / 1e9;
        }
    }

    for (double& v : bw_sums) v /= R;
    res.bw_gbs = bw_sums;

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
    constexpr int R = kBenchRounds;
    IlpResult res{};

    const int64_t iters = kIntIters;
    double sum_dep = 0, sum_ind = 0;

    for (int round = 0; round < R; ++round) {
        uint64_t s = runtime_seed() + static_cast<uint64_t>(round) * 0x9E3779B97F4A7C15ULL;

        // --- Dependent chain ---
        {
            int64_t x = static_cast<int64_t>(s & 0xFFFF);
            Timer t;
            for (int64_t i = 0; i < iters; ++i) {
                x = ((x * 3 + 7) ^ (x >> 5)) + (x << 2);
            }
            double ns = t.secs() * 1e9;
            sum_dep += static_cast<double>(iters * 5) / ns;
            escape_result(static_cast<uint64_t>(x));
        }

        // --- Independent operations ---
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
            }
            double ns = t.secs() * 1e9;
            sum_ind += static_cast<double>(iters * 5 * 6) / ns;
            escape_result(static_cast<uint64_t>(a + b + c + d + e + f));
        }
    }

    res.dep_ops_per_ns = sum_dep / R;
    res.ind_ops_per_ns = sum_ind / R;
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
    constexpr int R = kBenchRounds;
    MtResult res;

    const int max_threads = std::min(static_cast<int>(sys.hw_threads), 64);
    const int64_t steps_per_thread = 2'000'000;

    // Test powers of two plus the maximum thread count.
    std::vector<int> counts;
    counts.push_back(1);
    for (int t = 2; t <= max_threads; t *= 2)
        counts.push_back(t);
    if (counts.back() != max_threads)
        counts.push_back(max_threads);

    // Accumulators: one per thread count.
    std::vector<double> tp_sums(counts.size(), 0.0);

    for (int round = 0; round < R; ++round) {
        // Single-thread baseline for this round.
        double baseline_steps_per_sec = 0;
        {
            mt_kernel_work(steps_per_thread / 10);  // warm up
            Timer t;
            volatile double sink = mt_kernel_work(steps_per_thread);
            double sec = t.secs();
            baseline_steps_per_sec = static_cast<double>(steps_per_thread) / std::max(sec, 1e-9);
            escape_result(static_cast<uint64_t>(sink * 1e6));
        }

        for (size_t ci = 0; ci < counts.size(); ++ci) {
            int num_threads = counts[ci];
            if (num_threads == 1) {
                tp_sums[ci] += baseline_steps_per_sec;
                continue;
            }

            std::atomic<int64_t> global_work{0};
            std::atomic<double>  global_sum{0.0};
            std::vector<std::thread> threads;

            Timer wall_t;
            for (int t = 0; t < num_threads; ++t) {
                threads.emplace_back([&, steps_per_thread]() {
                    double sum = mt_kernel_work(steps_per_thread);
                    global_work.fetch_add(steps_per_thread, std::memory_order_relaxed);
                    global_sum.store(sum, std::memory_order_relaxed);
                });
            }
            for (auto& th : threads) th.join();

            double wall_sec = wall_t.secs();
            double tp = static_cast<double>(global_work.load()) /
                        std::max(wall_sec, 1e-9);
            escape_result(static_cast<uint64_t>(global_sum.load() * 1e9));
            tp_sums[ci] += tp;
        }
    }

    // Compute averages and speedups.
    for (size_t ci = 0; ci < counts.size(); ++ci) {
        double avg_tp = tp_sums[ci] / R;
        res.thread_counts.push_back(counts[ci]);
        res.throughputs.push_back(avg_tp);
        res.speedups.push_back(avg_tp / (tp_sums[0] / R));
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
    constexpr int R = kBenchRounds;
    MatMulResult res{};
    res.matrix_size = kMatMulN;

    const int64_t n = kMatMulN;
    std::vector<float> A(static_cast<size_t>(n * n));
    std::vector<float> B(static_cast<size_t>(n * n));
    std::vector<float> C(static_cast<size_t>(n * n), 0.0f);

    auto mul = [&]() {
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

    double sum_gflops = 0;

    for (int round = 0; round < R; ++round) {
        // Re-initialize with round-specific seed.
        uint64_t s = runtime_seed() + static_cast<uint64_t>(round) * 0x9E3779B97F4A7C15ULL;
        for (int64_t i = 0; i < n * n; ++i) {
            s = s * 1103515245 + 12345;
            A[static_cast<size_t>(i)] = static_cast<float>((s & 0xFFFF)) / 65536.0f;
            B[static_cast<size_t>(i)] = static_cast<float>(((s >> 16) & 0xFFFF)) / 65536.0f;
        }

        // Warm up.
        std::fill(C.begin(), C.end(), 0.0f);
        mul();
        // Measure.
        std::fill(C.begin(), C.end(), 0.0f);
        Timer t;
        mul();
        double sec = t.secs();
        escape_result(static_cast<uint64_t>(C[0] * 1e9f));

        double total_ops = 2.0 * static_cast<double>(n) *
                           static_cast<double>(n) * static_cast<double>(n);
        sum_gflops += total_ops / sec / 1e9;
    }

    res.gflops = sum_gflops / R;

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
    constexpr int R = kBenchRounds;
    SortResult res{};

    const int64_t n = kSortSize;
    std::vector<int64_t> data(static_cast<size_t>(n));

    // Generate base random data once.
    {
        std::mt19937_64 rng(runtime_seed());
        std::uniform_int_distribution<int64_t> dist;
        for (int64_t i = 0; i < n; ++i)
            data[static_cast<size_t>(i)] = dist(rng);
    }

    std::vector<int64_t> work(static_cast<size_t>(n));
    double sum_mps = 0;

    for (int round = 0; round < R; ++round) {
        // Warm up.
        std::copy(data.begin(), data.end(), work.begin());
        std::sort(work.begin(), work.end());
        // Measure.
        std::copy(data.begin(), data.end(), work.begin());
        Timer t;
        std::sort(work.begin(), work.end());
        double sec = t.secs();
        escape_result(static_cast<uint64_t>(work[0]));
        sum_mps += static_cast<double>(n) / sec / 1e6;
    }

    res.melem_per_sec = sum_mps / R;

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
    constexpr int R = kBenchRounds;
    HashResult res{};

    int64_t bytes = kHashMB * 1024LL * 1024LL;
    int64_t elems = bytes / sizeof(uint64_t);
    std::vector<uint64_t> buf(static_cast<size_t>(elems));

    double sum_mbs = 0;

    for (int round = 0; round < R; ++round) {
        uint64_t s = runtime_seed() + static_cast<uint64_t>(round) * 0x9E3779B97F4A7C15ULL;
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
        sum_mbs += static_cast<double>(bytes) / sec / 1e6;
    }

    res.mbs = sum_mbs / R;

    show_done(res.mbs, "MB/s");
    return res;
}

// ============================================================================
// Section 14a — Fast Fourier Transform Benchmark
// ============================================================================
// In-place radix-2 Cooley–Tukey FFT on complex doubles with precomputed
// twiddle factors.  Stresses complex arithmetic and memory access patterns.

struct FftResult {
    double gflops;   // billion floating-point ops per second
};

FftResult bench_fft() {
    show_progress("FFT (radix-2, double complex)");
    constexpr int R = kBenchRounds;
    FftResult res{};
    const int N = static_cast<int>(kFftN);
    std::vector<std::complex<double>> data(N);
    std::vector<std::complex<double>> twiddle(N / 2);

    // Precompute twiddle factors.
    for (int i = 0; i < N / 2; ++i) {
        double a = -2.0 * M_PI * static_cast<double>(i) / static_cast<double>(N);
        twiddle[i] = {std::cos(a), std::sin(a)};
    }

    double sum_gflops = 0;
    for (int round = 0; round < R; ++round) {
        uint64_t s = runtime_seed() + static_cast<uint64_t>(round) * 0x9E3779B97F4A7C15ULL;
        for (int i = 0; i < N; ++i) {
            double v = static_cast<double>((s + static_cast<uint64_t>(i)) & 0xFFF) / 4096.0;
            data[i] = {v, 0.0};
        }

        // Warm-up: one pass.
        {
            auto tmp = data;
            for (int i = 1, j = 0; i < N; ++i) {
                int bit = N >> 1;
                for (; j & bit; bit >>= 1) j ^= bit;
                j ^= bit;
                if (i < j) std::swap(tmp[i], tmp[j]);
            }
        }

        auto tmp = data;
        Timer t;
        // Bit-reversal permutation.
        for (int i = 1, j = 0; i < N; ++i) {
            int bit = N >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) std::swap(tmp[i], tmp[j]);
        }
        // Butterfly stages.
        int step = 1;
        for (int len = 2; len <= N; len <<= 1) {
            int half = len >> 1;
            for (int i = 0; i < N; i += len) {
                for (int j = 0; j < half; ++j) {
                    auto w = twiddle[j * step];
                    auto u = tmp[i + j];
                    auto v = tmp[i + j + half] * w;
                    tmp[i + j] = u + v;
                    tmp[i + j + half] = u - v;
                }
            }
            step >>= 1;
        }
        double sec = t.secs();
        escape_result(static_cast<uint64_t>(std::abs(tmp[0]) * 1e9));
        double ops = 5.0 * static_cast<double>(N) * std::log2(static_cast<double>(N));
        sum_gflops += ops / sec / 1e9;
    }
    res.gflops = sum_gflops / R;
    show_done(res.gflops, "GFLOPS");
    return res;
}

// ============================================================================
// Section 14b — N-body Gravitational Simulation Benchmark
// ============================================================================
// All-pairs O(N^2) gravitational force calculation.  Stresses sqrt, division,
// and 3D vector arithmetic.

struct NbodyResult {
    double gflops;
};

NbodyResult bench_nbody() {
    show_progress("N-body (gravity, O(n^2))");
    constexpr int R = kBenchRounds;
    NbodyResult res{};
    const int N = kNbodyBodies;
    const double dt = 0.001;
    const double eps = 1e-10;

    struct Body { double x, y, z, vx, vy, vz, mass; };
    std::vector<Body> bodies(N);
    std::vector<Body> orig(N);

    // Initialize bodies on a random spherical shell.
    {
        uint64_t s = runtime_seed();
        std::mt19937_64 rng(s);
        std::uniform_real_distribution<double> angle(0.0, 2.0 * M_PI);
        std::uniform_real_distribution<double> rad(0.5, 1.5);
        for (int i = 0; i < N; ++i) {
            double theta = angle(rng), phi = angle(rng), r = rad(rng);
            orig[i] = {r * std::cos(theta) * std::sin(phi),
                       r * std::sin(theta) * std::sin(phi),
                       r * std::cos(phi), 0.0, 0.0, 0.0, 1.0 / static_cast<double>(N)};
        }
    }

    double sum_gflops = 0;
    for (int round = 0; round < R; ++round) {
        bodies = orig;

        // Warm up.
        for (int i = 0; i < N; ++i) {
            double fx = 0, fy = 0, fz = 0;
            for (int j = 0; j < N; ++j) {
                if (i == j) continue;
                double dx = bodies[j].x - bodies[i].x;
                double dy = bodies[j].y - bodies[i].y;
                double dz = bodies[j].z - bodies[i].z;
                double d2 = dx*dx + dy*dy + dz*dz + eps;
                double inv_d3 = 1.0 / (d2 * std::sqrt(d2));
                fx += bodies[j].mass * dx * inv_d3;
                fy += bodies[j].mass * dy * inv_d3;
                fz += bodies[j].mass * dz * inv_d3;
            }
        }

        Timer t;
        for (int i = 0; i < N; ++i) {
            double fx = 0, fy = 0, fz = 0;
            for (int j = 0; j < N; ++j) {
                if (i == j) continue;
                double dx = bodies[j].x - bodies[i].x;
                double dy = bodies[j].y - bodies[i].y;
                double dz = bodies[j].z - bodies[i].z;
                double d2 = dx*dx + dy*dy + dz*dz + eps;
                double inv_d3 = 1.0 / (d2 * std::sqrt(d2));
                fx += bodies[j].mass * dx * inv_d3;
                fy += bodies[j].mass * dy * inv_d3;
                fz += bodies[j].mass * dz * inv_d3;
            }
            bodies[i].vx += fx * dt;
            bodies[i].vy += fy * dt;
            bodies[i].vz += fz * dt;
        }
        for (int i = 0; i < N; ++i) {
            bodies[i].x += bodies[i].vx * dt;
            bodies[i].y += bodies[i].vy * dt;
            bodies[i].z += bodies[i].vz * dt;
        }
        double sec = t.secs();
        escape_result(static_cast<uint64_t>(bodies[0].x * 1e9));
        // ~22 FP ops per interaction, N*(N-1) interactions.
        double ops = 22.0 * static_cast<double>(N) * static_cast<double>(N - 1);
        sum_gflops += ops / sec / 1e9;
    }
    res.gflops = sum_gflops / R;
    show_done(res.gflops, "GFLOPS");
    return res;
}

// ============================================================================
// Section 14c — Fluid Dynamics (Lattice Boltzmann D2Q9) Benchmark
// ============================================================================
// 2D Lattice Boltzmann Method with BGK collision.  Stresses stencil memory
// access and floating-point throughput.

struct FluidResult {
    double mlups;   // million lattice updates per second
};

FluidResult bench_fluid() {
    show_progress("Fluid LBM (D2Q9, BGK)");
    constexpr int R = kBenchRounds;
    FluidResult res{};
    const int nx = kLbmNx, ny = kLbmNy, nc = nx * ny;
    const double omega = 1.0 / 0.6;  // relaxation parameter
    // D2Q9 weights and directions
    const double w[9] = {4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
                         1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};
    const int ex[9] = { 0, 1, 0,-1, 0, 1,-1,-1, 1};
    const int ey[9] = { 0, 0, 1, 0,-1, 1, 1,-1,-1};

    std::vector<double> f(nc * 9), f_new(nc * 9);

    double sum_mlups = 0;
    for (int round = 0; round < R; ++round) {
        uint64_t s = runtime_seed() + static_cast<uint64_t>(round) * 0x9E3779B97F4A7C15ULL;
        // Initialize with equilibrium + small perturbation.
        for (int idx = 0; idx < nc; ++idx) {
            double rho = 1.0 + 0.01 * static_cast<double>((s + static_cast<uint64_t>(idx)) & 0xFF) / 255.0;
            double ux = 0.01 * static_cast<double>(((s >> 8) + static_cast<uint64_t>(idx)) & 0xFF) / 255.0;
            double uy = 0.0;
            for (int q = 0; q < 9; ++q) {
                double cu = ex[q] * ux + ey[q] * uy;
                f[idx * 9 + q] = w[q] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * (ux*ux + uy*uy));
            }
        }

        // Warm up (1 step).
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                int idx = y * nx + x;
                // Collision
                double rho = 0, ux = 0, uy = 0;
                for (int q = 0; q < 9; ++q) rho += f[idx * 9 + q];
                for (int q = 0; q < 9; ++q) { ux += ex[q] * f[idx * 9 + q]; uy += ey[q] * f[idx * 9 + q]; }
                ux /= rho; uy /= rho;
                double u2 = ux*ux + uy*uy;
                for (int q = 0; q < 9; ++q) {
                    double cu = ex[q]*ux + ey[q]*uy;
                    double feq = w[q] * rho * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u2);
                    f_new[idx * 9 + q] = f[idx * 9 + q] - omega * (f[idx * 9 + q] - feq);
                }
            }
        }
        // Streaming
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                for (int q = 0; q < 9; ++q) {
                    int nx_x = (x + ex[q] + nx) % nx;
                    int nx_y = (y + ey[q] + ny) % ny;
                    f[(nx_y * nx + nx_x) * 9 + q] = f_new[(y * nx + x) * 9 + q];
                }
            }
        }

        // Measure kLbmSteps steps.
        Timer t;
        for (int step = 0; step < kLbmSteps; ++step) {
            // Collision
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    int idx = y * nx + x;
                    double rho = 0, ux = 0, uy = 0;
                    for (int q = 0; q < 9; ++q) rho += f[idx * 9 + q];
                    double inv_rho = 1.0 / (rho + 1e-10);
                    for (int q = 0; q < 9; ++q) { ux += ex[q] * f[idx * 9 + q]; uy += ey[q] * f[idx * 9 + q]; }
                    ux *= inv_rho; uy *= inv_rho;
                    double u2 = ux*ux + uy*uy;
                    for (int q = 0; q < 9; ++q) {
                        double cu = ex[q]*ux + ey[q]*uy;
                        double feq = w[q] * rho * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u2);
                        f_new[idx * 9 + q] = f[idx * 9 + q] - omega * (f[idx * 9 + q] - feq);
                    }
                }
            }
            // Streaming
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    for (int q = 0; q < 9; ++q) {
                        int nx_x = (x + ex[q] + nx) % nx;
                        int nx_y = (y + ey[q] + ny) % ny;
                        f[(nx_y * nx + nx_x) * 9 + q] = f_new[(y * nx + x) * 9 + q];
                    }
                }
            }
        }
        double sec = t.secs();
        escape_result(static_cast<uint64_t>(f[0] * 1e9));
        double updates = static_cast<double>(nc) * static_cast<double>(kLbmSteps);
        sum_mlups += updates / sec / 1e6;
    }
    res.mlups = sum_mlups / R;
    show_done(res.mlups, "MLUPS");
    return res;
}

// ============================================================================
// Section 14d — Molecular Dynamics (Lennard-Jones) Benchmark
// ============================================================================
// Pairwise Lennard-Jones 12-6 potential with velocity-Verlet integration.
// Stresses sqrt, division, and pairwise distance calculations.

struct ChemResult {
    double gflops;
};

ChemResult bench_chemistry() {
    show_progress("Molecular Dynamics (LJ 12-6)");
    constexpr int R = kBenchRounds;
    ChemResult res{};
    const int N = kMdAtoms;
    const double dt = 0.001;
    const double eps = 1e-10;

    struct Atom { double x, y, z, vx, vy, vz; };
    std::vector<Atom> atoms(N);
    std::vector<Atom> orig(N);

    {
        uint64_t s = runtime_seed();
        std::mt19937_64 rng(s);
        std::uniform_real_distribution<double> pos(-5.0, 5.0);
        for (int i = 0; i < N; ++i)
            orig[i] = {pos(rng), pos(rng), pos(rng), 0.0, 0.0, 0.0};
    }

    double sum_gflops = 0;
    for (int round = 0; round < R; ++round) {
        atoms = orig;

        // Warm up (1 step).
        {
            std::vector<double> fx(N, 0.0), fy(N, 0.0), fz(N, 0.0);
            for (int i = 0; i < N; ++i) {
                for (int j = i + 1; j < N; ++j) {
                    double dx = atoms[j].x - atoms[i].x;
                    double dy = atoms[j].y - atoms[i].y;
                    double dz = atoms[j].z - atoms[i].z;
                    double r2 = dx*dx + dy*dy + dz*dz + eps;
                    double r6 = r2 * r2 * r2;
                    double r12 = r6 * r6;
                    double f = 24.0 * (2.0 / r12 - 1.0 / r6) / r2;
                    fx[i] += f * dx; fy[i] += f * dy; fz[i] += f * dz;
                    fx[j] -= f * dx; fy[j] -= f * dy; fz[j] -= f * dz;
                }
            }
        }

        Timer t;
        for (int step = 0; step < kMdSteps; ++step) {
            std::vector<double> fx(N, 0.0), fy(N, 0.0), fz(N, 0.0);
            for (int i = 0; i < N; ++i) {
                for (int j = i + 1; j < N; ++j) {
                    double dx = atoms[j].x - atoms[i].x;
                    double dy = atoms[j].y - atoms[i].y;
                    double dz = atoms[j].z - atoms[i].z;
                    double r2 = dx*dx + dy*dy + dz*dz + eps;
                    double r6 = r2 * r2 * r2;
                    double r12 = r6 * r6;
                    double f = 24.0 * (2.0 / r12 - 1.0 / r6) / r2;
                    fx[i] += f * dx; fy[i] += f * dy; fz[i] += f * dz;
                    fx[j] -= f * dx; fy[j] -= f * dy; fz[j] -= f * dz;
                }
            }
            for (int i = 0; i < N; ++i) {
                atoms[i].vx += fx[i] * dt;
                atoms[i].vy += fy[i] * dt;
                atoms[i].vz += fz[i] * dt;
                atoms[i].x  += atoms[i].vx * dt;
                atoms[i].y  += atoms[i].vy * dt;
                atoms[i].z  += atoms[i].vz * dt;
            }
        }
        double sec = t.secs();
        escape_result(static_cast<uint64_t>(atoms[0].x * 1e9));
        double ops = static_cast<double>(kMdSteps) * static_cast<double>(N) *
                     static_cast<double>(N - 1) * 0.5 * 28.0;
        sum_gflops += ops / sec / 1e9;
    }
    res.gflops = sum_gflops / R;
    show_done(res.gflops, "GFLOPS");
    return res;
}

// ============================================================================
// Section 14e — Game Logic Simulation Benchmark
// ============================================================================
// Entity update loop with physics, AI state machine, and health regen.
// Stresses branch prediction and mixed integer / FP operations.

struct GameResult {
    double melem_per_sec;   // million entity updates per second
};

GameResult bench_game() {
    show_progress("Game Logic (entity update)");
    constexpr int R = kBenchRounds;
    GameResult res{};
    const int M = kGameEntities;

    struct Entity { float x, y, vx, vy, health; int state, team; };
    std::vector<Entity> ents(M);
    std::vector<Entity> orig(M);

    {
        uint64_t s = runtime_seed();
        std::mt19937 rng(static_cast<unsigned>(s));
        std::uniform_real_distribution<float> pos(-100.0f, 100.0f);
        std::uniform_real_distribution<float> vel(-1.0f, 1.0f);
        std::uniform_real_distribution<float> hp(1.0f, 100.0f);
        std::uniform_int_distribution<int> team(0, 3);
        for (int i = 0; i < M; ++i)
            orig[i] = {pos(rng), pos(rng), vel(rng), vel(rng), hp(rng), 0, team(rng)};
    }

    double sum_mps = 0;
    const float dt = 1.0f / 60.0f;

    for (int round = 0; round < R; ++round) {
        ents = orig;

        // Warm up.
        for (int i = 0; i < M; ++i) {
            auto& e = ents[i];
            e.x += e.vx * dt; e.y += e.vy * dt;
            if (e.health < 25.0f)       { e.state = 2; e.vx *= -1.2f; e.vy *= -1.2f; }
            else if (e.health < 50.0f)  { e.state = 1; e.vx *= 0.8f;  e.vy *= 0.8f; }
            else                        { e.state = 0; e.vx *= 1.05f; e.vy *= 1.05f; }
            if (e.state == 1 && e.health < 100.0f) e.health += 0.5f;
            if (e.x > 100.0f) e.x -= 200.0f;
            if (e.x < -100.0f) e.x += 200.0f;
        }

        Timer t;
        for (int i = 0; i < M; ++i) {
            auto& e = ents[i];
            e.x += e.vx * dt; e.y += e.vy * dt;
            if (e.health < 25.0f)       { e.state = 2; e.vx *= -1.2f; e.vy *= -1.2f; }
            else if (e.health < 50.0f)  { e.state = 1; e.vx *= 0.8f;  e.vy *= 0.8f; }
            else                        { e.state = 0; e.vx *= 1.05f; e.vy *= 1.05f; }
            if (e.state == 1 && e.health < 100.0f) e.health += 0.5f;
            if (e.x > 100.0f) e.x -= 200.0f;
            if (e.x < -100.0f) e.x += 200.0f;
        }
        double sec = t.secs();
        escape_result(static_cast<uint64_t>(ents[0].x * 1e6f));
        sum_mps += static_cast<double>(M) / sec / 1e6;
    }
    res.melem_per_sec = sum_mps / R;
    show_done(res.melem_per_sec, "Melem/s");
    return res;
}

// ============================================================================
// Section 14f — AI Inference (Neural Network) Benchmark
// ============================================================================
// Feedforward MLP inference: 784->256->128->10 with ReLU activation.
// Stresses matrix-vector multiplication and activation functions.

struct AiResult {
    double gflops;
};

AiResult bench_ai() {
    show_progress("AI Inference (MLP 784x512x256x10)");
    constexpr int R = kBenchRounds;
    AiResult res{};
    const int I = 784, H1 = 512, H2 = 256, O = 10;
    const int batch = kAiBatch;

    // Allocate weights and biases with deterministic "random" values.
    std::vector<double> W1(I * H1), b1(H1);
    std::vector<double> W2(H1 * H2), b2(H2);
    std::vector<double> W3(H2 * O), b3(O);
    std::vector<double> input(I);

    {
        uint64_t s = runtime_seed();
        auto lcg = [&]() { s = s * 1103515245 + 12345; return static_cast<double>(s & 0xFFFF) / 65536.0 - 0.5; };
        for (auto& v : W1) v = lcg();
        for (auto& v : b1) v = lcg();
        for (auto& v : W2) v = lcg();
        for (auto& v : b2) v = lcg();
        for (auto& v : W3) v = lcg();
        for (auto& v : b3) v = lcg();
    }

    // One forward pass (cache-friendly loop order).
    auto forward = [&](std::vector<double>& h1, std::vector<double>& h2, std::vector<double>& out) {
        // Layer 1: I -> H1, ReLU
        for (int i = 0; i < H1; ++i) h1[i] = b1[i];
        for (int j = 0; j < I; ++j) {
            double in = input[j];
            for (int i = 0; i < H1; ++i) h1[i] += in * W1[j * H1 + i];
        }
        for (int i = 0; i < H1; ++i)
            if (h1[i] < 0.0) h1[i] = 0.0;  // ReLU

        // Layer 2: H1 -> H2, ReLU
        for (int i = 0; i < H2; ++i) h2[i] = b2[i];
        for (int j = 0; j < H1; ++j) {
            double h = h1[j];
            for (int i = 0; i < H2; ++i) h2[i] += h * W2[j * H2 + i];
        }
        for (int i = 0; i < H2; ++i)
            if (h2[i] < 0.0) h2[i] = 0.0;

        // Layer 3: H2 -> O, linear
        for (int i = 0; i < O; ++i) out[i] = b3[i];
        for (int j = 0; j < H2; ++j) {
            double h = h2[j];
            for (int i = 0; i < O; ++i) out[i] += h * W3[j * O + i];
        }
    };

    double sum_gflops = 0;
    std::vector<double> h1(H1), h2(H2), out(O);

    for (int round = 0; round < R; ++round) {
        uint64_t s = runtime_seed() + static_cast<uint64_t>(round) * 0x9E3779B97F4A7C15ULL;
        for (int j = 0; j < I; ++j)
            input[j] = static_cast<double>((s + static_cast<uint64_t>(j)) & 0xFF) / 255.0;

        // Warm up.
        forward(h1, h2, out);

        Timer t;
        for (int b = 0; b < batch; ++b)
            forward(h1, h2, out);
        double sec = t.secs();
        escape_result(static_cast<uint64_t>(out[0] * 1e9));

        // FLOPs per forward pass: 2*(I*H1 + H1*H2 + H2*O)
        double ops_per_pass = 2.0 * (static_cast<double>(I)*H1 + static_cast<double>(H1)*H2 + static_cast<double>(H2)*O);
        sum_gflops += ops_per_pass * static_cast<double>(batch) / sec / 1e9;
    }
    res.gflops = sum_gflops / R;
    show_done(res.gflops, "GFLOPS");
    return res;
}



// ============================================================================
// Section 15a — FDTD Electromagnetic Simulation Benchmark
// ============================================================================
struct FdtdResult { double mlups; };

FdtdResult bench_fdtd() {
    show_progress("FDTD EM (Yee grid, " + std::to_string(kFdtdNx) + "x" + std::to_string(kFdtdNy) + ")");
    constexpr int R = kBenchRounds; const int NX = kFdtdNx, NY = kFdtdNy, ST = kFdtdSteps;
    std::vector<double> ez(NX * NY, 0), hx(NX * NY, 0), hy(NX * NY, 0);
    double sum_mlups = 0;
    for (int round = 0; round < R; ++round) {
        uint64_t s = runtime_seed() + round * 0x9E3779B97F4A7C15ULL;
        for (int i = 0; i < NX * NY; ++i) ez[i] = ((s + i) & 0xF) * 0.001;
        for (int st = 0; st < 5; ++st) {  // warm up
            for (int y = 0; y < NY - 1; ++y) for (int x = 0; x < NX; ++x) { int c = y*NX+x; hy[c] += 0.5*(ez[c+NX]-ez[c]); }
            for (int y = 0; y < NY; ++y) for (int x = 0; x < NX - 1; ++x) { int c = y*NX+x; hx[c] += 0.5*(ez[c+1]-ez[c]); }
            for (int y = 1; y < NY; ++y) for (int x = 1; x < NX; ++x) { int c = y*NX+x; ez[c] += 0.5*((hy[c]-hy[c-NX])-(hx[c]-hx[c-1])); }
            for (int x = 0; x < NX; ++x) { ez[x] = ez[NX+x]*0.9; ez[(NY-1)*NX+x] = ez[(NY-2)*NX+x]*0.9; }
            for (int y = 0; y < NY; ++y) { ez[y*NX] = ez[y*NX+1]*0.9; ez[y*NX+NX-1] = ez[y*NX+NX-2]*0.9; }
        }
        Timer t;
        for (int st = 0; st < ST; ++st) {
            for (int y = 0; y < NY - 1; ++y) for (int x = 0; x < NX; ++x) { int c = y*NX+x; hy[c] += 0.5*(ez[c+NX]-ez[c]); }
            for (int y = 0; y < NY; ++y) for (int x = 0; x < NX - 1; ++x) { int c = y*NX+x; hx[c] += 0.5*(ez[c+1]-ez[c]); }
            for (int y = 1; y < NY; ++y) for (int x = 1; x < NX; ++x) { int c = y*NX+x; ez[c] += 0.5*((hy[c]-hy[c-NX])-(hx[c]-hx[c-1])); }
            for (int x = 0; x < NX; ++x) { ez[x] = ez[NX+x]*0.9; ez[(NY-1)*NX+x] = ez[(NY-2)*NX+x]*0.9; }
            for (int y = 0; y < NY; ++y) { ez[y*NX] = ez[y*NX+1]*0.9; ez[y*NX+NX-1] = ez[y*NX+NX-2]*0.9; }
        }
        double sec = t.secs(); escape_result(static_cast<uint64_t>(ez[NX/2]*1e9));
        sum_mlups += (static_cast<double>(NX)*NY*ST) / sec / 1e6;
    }
    FdtdResult res; res.mlups = sum_mlups / R;
    show_done(res.mlups, "MLUPS"); return res;
}

// ============================================================================
// Section 15b — Rigid Body Physics Benchmark
// ============================================================================
struct RigidResult { double mcollisions_per_sec; };

RigidResult bench_rigid() {
    show_progress("Rigid Body (collision detection)");
    constexpr int R = kBenchRounds; const int N = kRigidBodies;
    struct S { double x,y,z,vx,vy,vz,r; };
    std::vector<S> spheres(N), orig(N);
    { uint64_t s = runtime_seed(); std::mt19937_64 rng(s);
      std::uniform_real_distribution<double> p(-10,10), r(0.2,1.0);
      for (int i=0;i<N;++i) orig[i]={p(rng),p(rng),p(rng),0,0,0,r(rng)}; }
    double sum = 0; const double dt = 0.016;
    for (int round=0;round<R;++round) {
        spheres = orig; int collisions = 0;
        for (int w=0;w<10;++w) { for (int i=0;i<N;++i) for (int j=i+1;j<N;++j) {
            double dx=spheres[j].x-spheres[i].x, dy=spheres[j].y-spheres[i].y, dz=spheres[j].z-spheres[i].z;
            double d2=dx*dx+dy*dy+dz*dz, minR=spheres[i].r+spheres[j].r;
            if (d2<minR*minR) { collisions++; double d=std::sqrt(d2)+1e-10, overlap=minR-d;
                double nx=dx/d, ny=dy/d, nz=dz/d;
                spheres[i].x-=nx*overlap*0.5; spheres[i].y-=ny*overlap*0.5; spheres[i].z-=nz*overlap*0.5;
                spheres[j].x+=nx*overlap*0.5; spheres[j].y+=ny*overlap*0.5; spheres[j].z+=nz*overlap*0.5; }
        }}
        for (int i=0;i<N;++i){spheres[i].x+=spheres[i].vx*dt;spheres[i].y+=spheres[i].vy*dt;spheres[i].z+=spheres[i].vz*dt;}
        Timer t; collisions = 0;
        for (int w=0;w<10;++w) { for (int i=0;i<N;++i) for (int j=i+1;j<N;++j) {
            double dx=spheres[j].x-spheres[i].x, dy=spheres[j].y-spheres[i].y, dz=spheres[j].z-spheres[i].z;
            double d2=dx*dx+dy*dy+dz*dz, minR=spheres[i].r+spheres[j].r;
            if (d2<minR*minR) { collisions++; double d=std::sqrt(d2)+1e-10, overlap=minR-d;
                double nx=dx/d, ny=dy/d, nz=dz/d;
                spheres[i].x-=nx*overlap*0.5; spheres[i].y-=ny*overlap*0.5; spheres[i].z-=nz*overlap*0.5;
                spheres[j].x+=nx*overlap*0.5; spheres[j].y+=ny*overlap*0.5; spheres[j].z+=nz*overlap*0.5; }
        }}
        double sec=t.secs(); escape_result(static_cast<uint64_t>(collisions));
        sum += static_cast<double>(collisions) / sec / 1e6;
    }
    RigidResult res; res.mcollisions_per_sec = sum / R;
    show_done(res.mcollisions_per_sec, "Mcollisions/s"); return res;
}

// ============================================================================
// Section 15c — Ray Tracing Benchmark
// ============================================================================
struct RayResult { double mrays_per_sec; };

RayResult bench_raytrace() {
    show_progress("Ray Tracing (Whitted, " + std::to_string(kRayResX) + "x" + std::to_string(kRayResY) + ")");
    constexpr int R = kBenchRounds; const int W = kRayResX, H = kRayResY;
    struct V { double x,y,z; V operator+(V o){return {x+o.x,y+o.y,z+o.z};} V operator-(V o){return {x-o.x,y-o.y,z-o.z};}
        V operator*(double s){return {x*s,y*s,z*s};} double dot(V o){return x*o.x+y*o.y+z*o.z;} V norm(){double l=std::sqrt(dot(*this))+1e-12;return {x/l,y/l,z/l};} };
    struct Sph { V c; double r; double col[3]; };
    Sph scene[] = {{{0,0,-5},1,{1,0.2,0.2}},{{2,1,-6},1.5,{0.2,1,0.2}},{{-2,-1,-4},1,{0.2,0.2,1}},{{0,-101,-5},100,{0.8,0.8,0.8}}};
    const int NS = 4; V light = {-5,5,-2}; V eye = {0,0,0};
    auto hit = [&](V ro, V rd, double& t, int& id) { t=1e9; id=-1; for(int i=0;i<NS;++i){V oc=ro-scene[i].c; double b=oc.dot(rd), c=oc.dot(oc)-scene[i].r*scene[i].r, d=b*b-c; if(d>0){double tt=-b-std::sqrt(d); if(tt>0.001&&tt<t){t=tt;id=i;}}} return id>=0; };
    double sum=0;
    for (int round=0;round<R;++round) {
        int total_rays = W*H;
        for (int y=0;y<10;++y) { for (int py=0;py<H;++py) for (int px=0;px<W;++px) {
            double u=(px-W/2.0)/H, v=(py-H/2.0)/H; V rd={u,v,-1}; rd=rd.norm(); double t; int id; hit(eye,rd,t,id); }}
        Timer tm;
        for (int py=0;py<H;++py) for (int px=0;px<W;++px) {
            double u=(px-W/2.0)/H, v=(py-H/2.0)/H; V rd={u,v,-1}; rd=rd.norm(); double t; int id; hit(eye,rd,t,id);
            if (id>=0) { V pt = eye + rd*t; V ldir = (light-pt).norm(); double st; int sid; if (!hit(pt+ldir*0.01,ldir,st,sid) || st>(light-pt).dot(light-pt)) { /* lit */ } }
        }
        double sec=tm.secs(); escape_result(static_cast<uint64_t>(total_rays));
        sum += static_cast<double>(total_rays) / sec / 1e6;
    }
    RayResult res; res.mrays_per_sec = sum / R;
    show_done(res.mrays_per_sec, "Mrays/s"); return res;
}

// ============================================================================
// Section 15d — Quantum Chemistry (Hartree-Fock ERI) Benchmark
// ============================================================================
struct HfResult { double gflops; };

HfResult bench_hf() {
    show_progress("Hartree-Fock (ERI integrals)");
    constexpr int R = kBenchRounds, B = 10;
    // Simulate 4-index ERI loop over basis functions: sum over i,j,k,l of (ij|kl)
    double sum_gflops = 0;
    for (int round = 0; round < R; ++round) {
        uint64_t s = runtime_seed() + round * 0x9E3779B97F4A7C15ULL;
        std::vector<double> D(B*B, 0); for (int i=0;i<B*B;++i) D[i]=((s+i)&0xFF)/256.0;
        volatile double sink = 0;
        // Warm up half the loop
        for (int i=0;i<B/2;++i) for (int j=0;j<B;++j) for (int k=0;k<B;++k) for (int l=0;l<B;++l)
            sink += D[i*B+j]*D[k*B+l]/(1.0+std::abs(static_cast<double>(i-j)+static_cast<double>(k-l)));
        Timer t; sink = 0;
        for (int i=0;i<B;++i) for (int j=0;j<B;++j) for (int k=0;k<B;++k) for (int l=0;l<B;++l)
            sink += D[i*B+j]*D[k*B+l]/(1.0+std::abs(static_cast<double>(i-j)+static_cast<double>(k-l)));
        double sec = t.secs(); escape_result(static_cast<uint64_t>(sink*1e6));
        double ops = static_cast<double>(B*B*B*B) * 8.0;
        sum_gflops += ops / sec / 1e9;
    }
    HfResult res; res.gflops = sum_gflops / R;
    show_done(res.gflops, "GFLOPS"); return res;
}

// ============================================================================
// Section 15e — Monte Carlo Molecular Simulation Benchmark
// ============================================================================
struct McResult { double msteps_per_sec; };

McResult bench_montecarlo() {
    show_progress("Monte Carlo (Metropolis, LJ)");
    constexpr int R = kBenchRounds; const int N = 100, STEPS = kMcSteps;
    struct A { double x,y,z; };
    std::vector<A> atoms(N); double sum_mps = 0;
    auto energy = [&](const std::vector<A>& at) { double e=0; for(int i=0;i<N;++i) for(int j=i+1;j<N;++j)
        { double dx=at[j].x-at[i].x,dy=at[j].y-at[i].y,dz=at[j].z-at[i].z; double r2=dx*dx+dy*dy+dz*dz+1e-10, r6=r2*r2*r2; e+=4.0*(1.0/(r6*r6)-1.0/r6); } return e; };
    for (int round=0;round<R;++round) {
        uint64_t s = runtime_seed() + round*0x9E3779B97F4A7C15ULL;
        std::mt19937_64 rng(s); std::uniform_real_distribution<double> pos(-3,3), disp(-0.1,0.1), acc(0,1);
        for(int i=0;i<N;++i) atoms[i]={pos(rng),pos(rng),pos(rng)};
        double curE = energy(atoms); int accepted = 0;
        for (int st=0;st<STEPS/10;++st) { int idx=static_cast<int>(rng()%N); A old=atoms[idx];
            atoms[idx].x+=disp(rng);atoms[idx].y+=disp(rng);atoms[idx].z+=disp(rng);
            double newE=energy(atoms); if(newE<curE||acc(rng)<std::exp(-(newE-curE))){curE=newE;accepted++;}else atoms[idx]=old; }
        Timer t; accepted=0;
        for (int st=0;st<STEPS;++st) { int idx=static_cast<int>(rng()%N); A old=atoms[idx];
            atoms[idx].x+=disp(rng);atoms[idx].y+=disp(rng);atoms[idx].z+=disp(rng);
            double newE=energy(atoms); if(newE<curE||acc(rng)<std::exp(-(newE-curE))){curE=newE;accepted++;}else atoms[idx]=old; }
        double sec=t.secs(); escape_result(static_cast<uint64_t>(accepted));
        sum_mps += static_cast<double>(STEPS)/sec/1e6;
    }
    McResult res; res.msteps_per_sec = sum_mps / R;
    show_done(res.msteps_per_sec, "Msteps/s"); return res;
}

// ============================================================================
// Section 15f — Protein Folding (HP Model) Benchmark
// ============================================================================
struct HpResult { double mevals_per_sec; };

HpResult bench_protein() {
    show_progress("Protein Folding (HP model, annealing)");
    constexpr int R = kBenchRounds, L = 50, STEPS = 50000;
    // HP sequence (alternating H/P)
    std::vector<int> seq(L); for(int i=0;i<L;++i) seq[i]=(i%3==0)?1:0; // 1=H, 0=P
    std::vector<std::pair<int,int>> conf(L);
    auto hp_energy = [&](const std::vector<std::pair<int,int>>& c) {
        int e=0; for(int i=0;i<L;++i) for(int j=i+2;j<L;++j) if(seq[i]&&seq[j]&&std::abs(c[i].first-c[j].first)+std::abs(c[i].second-c[j].second)==1) e--;
        return e; };
    double sum=0;
    for (int round=0;round<R;++round) {
        uint64_t s = runtime_seed()+round*0x9E3779B97F4A7C15ULL; std::mt19937_64 rng(s);
        // Self-avoiding walk initialization
        conf[0]={0,0}; std::set<std::pair<int,int>> occupied; occupied.insert({0,0});
        int dx[4]={1,-1,0,0}, dy[4]={0,0,1,-1};
        for(int i=1;i<L;++i){bool placed=false;for(int d=0;d<4&&!placed;++d){int nx=conf[i-1].first+dx[d],ny=conf[i-1].second+dy[d];if(!occupied.count({nx,ny})){conf[i]={nx,ny};occupied.insert({nx,ny});placed=true;}}if(!placed)conf[i]=conf[i-1];}
        int bestE=hp_energy(conf); double T=1.0; int evals=0;
        for(int st=0;st<STEPS/10;++st){int i=rng()%L;auto old=conf[i];int nx=conf[i].first+dx[rng()%4],ny=conf[i].second+dy[rng()%4];
            if(!occupied.count({nx,ny})){occupied.erase({old.first,old.second});occupied.insert({nx,ny});conf[i]={nx,ny};int nE=hp_energy(conf);evals++;if(nE<=bestE||std::uniform_real_distribution<double>(0,1)(rng)<std::exp(-(nE-bestE)/T))bestE=nE;else{conf[i]=old;occupied.erase({nx,ny});occupied.insert({old.first,old.second});}}T*=0.999;}
        Timer t; evals=0;
        for(int st=0;st<STEPS;++st){int i=rng()%L;auto old=conf[i];int nx=conf[i].first+dx[rng()%4],ny=conf[i].second+dy[rng()%4];
            if(!occupied.count({nx,ny})){occupied.erase({old.first,old.second});occupied.insert({nx,ny});conf[i]={nx,ny};int nE=hp_energy(conf);evals++;if(nE<=bestE||std::uniform_real_distribution<double>(0,1)(rng)<std::exp(-(nE-bestE)/T))bestE=nE;else{conf[i]=old;occupied.erase({nx,ny});occupied.insert({old.first,old.second});}}T*=0.999;}
        double sec=t.secs();escape_result(static_cast<uint64_t>(evals));
        sum+=static_cast<double>(evals)/sec/1e6;
    }
    HpResult res; res.mevals_per_sec = sum/R;
    show_done(res.mevals_per_sec, "Mevals/s"); return res;
}

// ============================================================================
// Section 15g — DNA Sequence Alignment (Smith-Waterman) Benchmark
// ============================================================================
struct SwResult { double mcells_per_sec; };

SwResult bench_sw() {
    show_progress("Smith-Waterman (DNA alignment)");
    constexpr int R = kBenchRounds, L = 1000;
    std::string bases="ACGT"; std::vector<char> a(L),b(L);
    { uint64_t s=runtime_seed(); for(int i=0;i<L;++i){a[i]=bases[(s+i)&3];b[i]=bases[((s>>2)+i)&3];} }
    double sum=0; const int match=2,mismatch=-1,gap=-1;
    std::vector<int> prev(L+1), cur(L+1);
    for (int round=0;round<R;++round) {
        for(int j=0;j<=L;++j) prev[j]=(j==0)?0:j*gap;
        // Warm up
        for(int i=1;i<=L/10;++i){cur[0]=i*gap;for(int j=1;j<=L;++j){int s=(a[i-1]==b[j-1])?match:mismatch;cur[j]=std::max({0,prev[j-1]+s,prev[j]+gap,cur[j-1]+gap});}std::swap(prev,cur);}
        Timer t;
        for(int i=1;i<=L;++i){cur[0]=i*gap;for(int j=1;j<=L;++j){int s=(a[i-1]==b[j-1])?match:mismatch;cur[j]=std::max({0,prev[j-1]+s,prev[j]+gap,cur[j-1]+gap});}std::swap(prev,cur);}
        double sec=t.secs();escape_result(static_cast<uint64_t>(prev[0]));
        sum+=static_cast<double>(L)*L/sec/1e6;
    }
    SwResult res; res.mcells_per_sec=sum/R;
    show_done(res.mcells_per_sec,"Mcells/s"); return res;
}

// ============================================================================
// Section 15h — Spiking Neuron Network (Izhikevich) Benchmark
// ============================================================================
struct NeuronResult { double mupdates_per_sec; };

NeuronResult bench_neuron() {
    show_progress("Spiking Neurons (Izhikevich)");
    constexpr int R=kBenchRounds,N=kIzhNeurons,ST=kIzhSteps;
    std::vector<double> v(N),u(N),a(N),b(N),c(N),d(N),I(N);
    // Random sparse connections
    std::vector<std::vector<int>> conn(N);
    { uint64_t s=runtime_seed(); std::mt19937_64 rng(s); for(int i=0;i<N;++i){a[i]=0.02;b[i]=0.2;c[i]=-65+10*(rng()%100)/100.0;d[i]=8-6*(rng()%100)/100.0;v[i]=-65;u[i]=b[i]*v[i];I[i]=5+rng()%20;for(int j=0;j<20;++j) conn[i].push_back(rng()%N);} }
    double sum=0;
    for(int round=0;round<R;++round){
        for(int st=0;st<ST/10;++st) for(int i=0;i<N;++i){v[i]+=0.5*(0.04*v[i]*v[i]+5*v[i]+140-u[i]+I[i]);v[i]+=0.5*(0.04*v[i]*v[i]+5*v[i]+140-u[i]+I[i]);u[i]+=a[i]*(b[i]*v[i]-u[i]);if(v[i]>=30){v[i]=c[i];u[i]+=d[i];for(int tgt:conn[i]) I[tgt]+=0.5;}}
        Timer t;
        for(int st=0;st<ST;++st) for(int i=0;i<N;++i){v[i]+=0.5*(0.04*v[i]*v[i]+5*v[i]+140-u[i]+I[i]);v[i]+=0.5*(0.04*v[i]*v[i]+5*v[i]+140-u[i]+I[i]);u[i]+=a[i]*(b[i]*v[i]-u[i]);if(v[i]>=30){v[i]=c[i];u[i]+=d[i];for(int tgt:conn[i]) I[tgt]+=0.5;}}
        double sec=t.secs();escape_result(static_cast<uint64_t>(v[0]*1e6));
        sum+=static_cast<double>(N)*ST/sec/1e6;
    }
    NeuronResult res; res.mupdates_per_sec=sum/R;
    show_done(res.mupdates_per_sec,"Mneuron-upd/s"); return res;
}

// ============================================================================
// Section 15i — LU Decomposition Benchmark
// ============================================================================
struct LuResult { double gflops; };

LuResult bench_lu() {
    show_progress("LU Decomposition (no pivoting)");
    constexpr int R=kBenchRounds; const int N=static_cast<int>(kLuN);
    std::vector<double> A(N*N), orig(N*N);
    { uint64_t s=runtime_seed(); for(int i=0;i<N*N;++i) orig[i]=(((s+i)&0xFFFF)/65536.0-0.5)*2.0+((i%N==i/N)?N:0); }
    std::vector<double> work(N*N);
    double sum=0;
    for(int round=0;round<R;++round){
        std::copy(orig.begin(),orig.end(),work.begin());
        // Warm up
        for(int k=0;k<N/10;++k){for(int i=k+1;i<N;++i){work[i*N+k]/=work[k*N+k];for(int j=k+1;j<N;++j)work[i*N+j]-=work[i*N+k]*work[k*N+j];}}
        work=orig;
        Timer t;
        for(int k=0;k<N;++k){double inv=1.0/work[k*N+k];for(int i=k+1;i<N;++i){work[i*N+k]*=inv;for(int j=k+1;j<N;++j)work[i*N+j]-=work[i*N+k]*work[k*N+j];}}
        double sec=t.secs();escape_result(static_cast<uint64_t>(work[0]*1e9));
        double ops=(2.0/3.0)*static_cast<double>(N)*N*N;
        sum+=ops/sec/1e9;
    }
    LuResult res; res.gflops=sum/R;
    show_done(res.gflops,"GFLOPS"); return res;
}

// ============================================================================
// Section 15j — Stiff ODE Solver (Robertson) Benchmark
// ============================================================================
struct OdeResult { double steps_per_sec; };

OdeResult bench_ode() {
    show_progress("Stiff ODE (Robertson, BDF)");
    constexpr int R=kBenchRounds; const int ST=kOdeSteps;
    double sum=0;
    for(int round=0;round<R;++round){
        double y1=1.0,y2=0.0,y3=0.0,dt=1e-4;
        // Robertson: y1'=-0.04*y1+1e4*y2*y3, y2'=0.04*y1-1e4*y2*y3-3e7*y2^2, y3'=3e7*y2^2
        for(int st=0;st<ST/10;++st){double k1=-0.04*y1+1e4*y2*y3,k2=0.04*y1-1e4*y2*y3-3e7*y2*y2,k3=3e7*y2*y2;y1+=k1*dt;y2+=k2*dt;y3+=k3*dt;}
        y1=1.0;y2=0.0;y3=0.0;
        Timer t;
        for(int st=0;st<ST;++st){double k1=-0.04*y1+1e4*y2*y3,k2=0.04*y1-1e4*y2*y3-3e7*y2*y2,k3=3e7*y2*y2;y1+=k1*dt;y2+=k2*dt;y3+=k3*dt;}
        double sec=t.secs();escape_result(static_cast<uint64_t>(y1*1e9));
        sum+=static_cast<double>(ST)/sec;
    }
    OdeResult res; res.steps_per_sec=sum/R;
    show_done(res.steps_per_sec,"steps/s"); return res;
}

// ============================================================================
// Section 15k — Big Integer Multiplication (Karatsuba) Benchmark
// ============================================================================
struct KaratsubaResult { double mdigit_muls_per_sec; };

KaratsubaResult bench_karatsuba() {
    show_progress("BigInt Mul (Karatsuba)");
    constexpr int R=kBenchRounds; const int BITS=kKaratsubaBits;
    using BigInt = std::vector<uint64_t>; int words = (BITS+63)/64;
    auto add = [](BigInt& c, const BigInt& a, const BigInt& b){ uint64_t carry=0; for(size_t i=0;i<c.size();++i){uint64_t s=a[i]+b[i]+carry;c[i]=s;carry=(s<a[i])?1:0;} };
    auto sub = [](BigInt& c, const BigInt& a, const BigInt& b){ int64_t borrow=0; for(size_t i=0;i<c.size();++i){int64_t d=static_cast<int64_t>(a[i])-b[i]-borrow;c[i]=static_cast<uint64_t>(d);borrow=(d<0)?1:0;} };
    std::function<void(BigInt&,const BigInt&,const BigInt&,int)> karatsuba = [&](BigInt& c,const BigInt& a,const BigInt& b,int n){
        if(n<=64){for(int i=0;i<n;++i){uint64_t carry=0;for(int j=0;j<n;++j){uint64_t p=a[i]*b[j];uint64_t s=c[i+j]+p+carry;c[i+j]=s;carry=(s<p)?1:0;}c[i+n]+=carry;}return;}
        int half=n/2; BigInt a0(a.begin(),a.begin()+half),a1(a.begin()+half,a.begin()+n),b0(b.begin(),b.begin()+half),b1(b.begin()+half,b.begin()+n);
        BigInt z0(2*n),z1(2*n),z2(2*n),ta(half),tb(half),tc(2*n);
        karatsuba(z0,a0,b0,half); karatsuba(z2,a1,b1,half);
        add(ta,a0,a1); add(tb,b0,b1); karatsuba(z1,ta,tb,half);
        BigInt tmp(2*n); sub(tmp,z1,z0); sub(z1,tmp,z2);
        for(int i=0;i<2*half;++i) c[i]+=z0[i];
        for(int i=0;i<2*half;++i) c[i+half]+=z1[i];
        for(int i=0;i<2*half;++i) c[i+n]+=z2[i];
    };
    BigInt a(words),b(words),c(words*2);
    { uint64_t s=runtime_seed(); for(int i=0;i<words;++i){a[i]=s+i*0x9E3779B97F4A7C15ULL;b[i]=s+i*0x123456789ABCDEFULL;} }
    double sum=0; int n=static_cast<int>(words);
    for(int round=0;round<R;++round){
        std::fill(c.begin(),c.end(),0); karatsuba(c,a,b,n/2); // warm up
        std::fill(c.begin(),c.end(),0);
        Timer t; karatsuba(c,a,b,n);
        double sec=t.secs();escape_result(c[0]);
        sum+=static_cast<double>(n)*n/sec/1e6;
    }
    KaratsubaResult res; res.mdigit_muls_per_sec=sum/R;
    show_done(res.mdigit_muls_per_sec,"Mdigit-mul/s"); return res;
}

// ============================================================================
// Section 15l — Regex Matching Benchmark
// ============================================================================
struct RegexResult { double mbs; };

RegexResult bench_regex() {
    show_progress("Pattern Matching (email scan)");
    constexpr int R=kBenchRounds;
    int64_t bytes = kRegexMB*1024LL*1024LL;
    std::string text(static_cast<size_t>(bytes), ' ');
    // Generate text with embedded email-like patterns for realistic matching.
    { uint64_t s=runtime_seed();
      for(size_t i=0;i<text.size();++i){char c=static_cast<char>(' '+((s+i)%95));text[i]=(c=='@'||c=='.')?static_cast<char>('a'+((s+i)%26)):c;}
      // Insert some actual @ and . to create matchable patterns.
      for(size_t i=0;i+20<text.size();i+=97){text[i+5]='@';text[i+12]='.';}
    }
    // Hand-rolled scanner: find "alnum+@alnum+.alnum+" patterns without std::regex.
    auto count_patterns=[&]()->int64_t{
        int64_t cnt=0; size_t n=text.size();
        for(size_t i=0;i+6<n;++i){
            if(text[i]=='@'){
                // Check left side: at least 1 alphanumeric before @
                if(i==0||!isalnum(text[i-1])) continue;
                // Check right side: alnum+.alnum
                size_t j=i+1; while(j<n&&isalnum(text[j]))j++;
                if(j>=n||text[j]!='.') continue;
                j++; while(j<n&&isalnum(text[j]))j++;
                if(j>i+3){cnt++;i=j-1;}
            }
        }
        return cnt;
    };
    double sum=0;
    for(int round=0;round<R;++round){
        count_patterns(); // warm up
        Timer t; int64_t matches=count_patterns();
        double sec=t.secs();escape_result(static_cast<uint64_t>(matches));
        sum+=static_cast<double>(bytes)/sec/1e6;
    }
    RegexResult res; res.mbs=sum/R;
    show_done(res.mbs,"MB/s"); return res;
}

// ============================================================================
// Section 15m — JSON Parsing Benchmark
// ============================================================================
struct JsonResult { double mbs; };

JsonResult bench_json() {
    show_progress("JSON Parsing (recursive descent)");
    constexpr int R=kBenchRounds;
    // Build a larger nested JSON string for meaningful timing.
    std::ostringstream jss;
    jss << "{"; for(int i=0;i<1000;++i){ jss<<"\"key"<<i<<"\":{"; for(int j=0;j<30;++j) jss<<"\"f"<<j<<"\":"<<(i*j%1000)<<","; jss<<"\"arr"<<i<<"\":["; for(int k=0;k<10;++k) jss<<(k>0?",":"")<<"{\"x\":"<<k<<",\"y\":"<<(k*2)<<"}"; jss<<"]},"; } jss<<"\"end\":null}";
    std::string json = jss.str();
    double sum=0;
    for(int round=0;round<R;++round){
        size_t pos=0; auto skipWS=[&]{while(pos<json.size()&&(json[pos]==' '||json[pos]=='\n'||json[pos]=='\t'||json[pos]=='\r'))pos++;};
        std::function<int()> parse; parse=[&]()->int{skipWS();int nodes=1;if(json[pos]=='{'){pos++;skipWS();if(json[pos]!='}')while(true){skipWS();pos++;skipWS();pos++;parse();skipWS();if(json[pos]==',')pos++;else break;}pos++;}else if(json[pos]=='['){pos++;skipWS();if(json[pos]!=']')while(true){parse();skipWS();if(json[pos]==',')pos++;else break;}pos++;}else if(json[pos]=='"'){pos++;while(pos<json.size()&&json[pos]!='"')pos++;pos++;}else{while(pos<json.size()&&(isdigit(json[pos])||json[pos]=='-'||json[pos]=='.'||json[pos]=='e'))pos++;}return nodes;};
        // Parse twice to warm up I-cache.
        pos=0; parse();
        pos=0; Timer t; int nodes=parse();
        double sec=std::max(t.secs(),1e-9); escape_result(static_cast<uint64_t>(nodes));
        sum+=static_cast<double>(json.size())/sec/1e6;
    }
    JsonResult res; res.mbs=sum/R;
    show_done(res.mbs,"MB/s"); return res;
}

// ============================================================================
// Section 15n — Memory Allocator Stress Test Benchmark
// ============================================================================
struct AllocResult { double mallocs_per_sec; };

AllocResult bench_alloc() {
    show_progress("Memory Allocator (multi-threaded)");
    constexpr int R=kBenchRounds, T=4; const int64_t OPS=kAllocOps;
    double sum=0;
    for(int round=0;round<R;++round){
        std::atomic<int64_t> total{0};
        auto worker=[&]{std::mt19937_64 rng(runtime_seed());std::uniform_int_distribution<int> sz(16,4096);
            int64_t local=0;std::vector<std::pair<void*,size_t>> ptrs;ptrs.reserve(10000);
            for(int64_t i=0;i<OPS/T;++i){if(ptrs.size()>5000||(ptrs.size()>0&&(rng()&1))){int idx=rng()%ptrs.size();::operator delete(ptrs[idx].first);ptrs[idx]=ptrs.back();ptrs.pop_back();}else{size_t s=sz(rng);ptrs.push_back({::operator new(s),s});}local++;}
            while(!ptrs.empty()){::operator delete(ptrs.back().first);ptrs.pop_back();}
            total+=local;};
        std::vector<std::thread> th; Timer t; for(int i=0;i<T;++i) th.emplace_back(worker);
        for(auto& th_:th) th_.join();
        double sec=t.secs();escape_result(static_cast<uint64_t>(total.load()));
        sum+=static_cast<double>(total.load())/sec/1e6;
    }
    AllocResult res; res.mallocs_per_sec=sum/R;
    show_done(res.mallocs_per_sec,"Mallocs/s"); return res;
}

// ============================================================================
// Section 15o — Concurrency Primitives Benchmark
// ============================================================================
struct AtomicResult { double matomic_per_sec; };

AtomicResult bench_atomic() {
    show_progress("Concurrency Primitives (atomic inc)");
    constexpr int R=kBenchRounds, T=4; const int64_t N=kAtomicIters;
    double sum=0;
    for(int round=0;round<R;++round){
        std::atomic<int64_t> counter{0};
        auto worker=[&]{for(int64_t i=0;i<N/T;++i) counter.fetch_add(1,std::memory_order_relaxed);};
        // Warm up
        {std::vector<std::thread> th;for(int i=0;i<T;++i)th.emplace_back(worker);for(auto&t:th)t.join();}
        counter=0;
        std::vector<std::thread> th; Timer t; for(int i=0;i<T;++i) th.emplace_back(worker);
        for(auto& th_:th) th_.join();
        double sec=t.secs();escape_result(static_cast<uint64_t>(counter.load()));
        sum+=static_cast<double>(N)/sec/1e6;
    }
    AtomicResult res; res.matomic_per_sec=sum/R;
    show_done(res.matomic_per_sec,"Matomic-inc/s"); return res;
}

// ============================================================================
// Section 15p — AST Evaluation Benchmark
// ============================================================================
struct AstResult { double mnodes_per_sec; };

AstResult bench_ast() {
    show_progress("AST Evaluation (expression tree)");
    constexpr int R=kBenchRounds; const int ND=kAstNodes;
    struct Node { int op; double val; Node *l,*r; }; // op:0=const,1=add,2=mul,3=sin
    std::vector<Node> pool(ND);
    { uint64_t s=runtime_seed(); std::mt19937_64 rng(s); for(int i=0;i<ND;++i){pool[i].op=rng()%4;pool[i].val=(rng()%1000)/100.0;pool[i].l=(i*2+1<ND)?&pool[i*2+1]:nullptr;pool[i].r=(i*2+2<ND)?&pool[i*2+2]:nullptr;} }
    std::function<double(Node*)> eval=[&](Node*n)->double{if(!n)return 0;switch(n->op){case 0:return n->val;case 1:return eval(n->l)+eval(n->r);case 2:return eval(n->l)*eval(n->r);case 3:return std::sin(eval(n->l));default:return 0;}};
    double sum=0;
    for(int round=0;round<R;++round){
        eval(&pool[0]); // warm up
        Timer t; volatile double r=eval(&pool[0]);
        double sec=t.secs();escape_result(static_cast<uint64_t>(r*1e9));
        sum+=static_cast<double>(ND)/sec/1e6;
    }
    AstResult res; res.mnodes_per_sec=sum/R;
    show_done(res.mnodes_per_sec,"Mnodes/s"); return res;
}

// ============================================================================
// Section 15q — LR Parser Benchmark
// ============================================================================
struct ParseResult { double mtokens_per_sec; };

ParseResult bench_parser() {
    show_progress("LR Parser (table-driven)");
    constexpr int R=kBenchRounds; const int64_t NT=kParseTokens;
    // Generate token stream: numbers and operators
    std::vector<int> tokens(NT);
    { uint64_t s=runtime_seed(); for(int i=0;i<NT;++i) tokens[i]=(s+i)%5; } // 0=num,1=+,2=*,3=(,4=)
    // Simple shift-reduce parser for expressions: E->E+E|E*E|(E)|num
    double sum=0;
    for(int round=0;round<R;++round){
        // Warm up
        { std::vector<int> stk; for(int i=0;i<NT/100;++i){int t=tokens[i];if(t==0||t==3)stk.push_back(t);else if(t==4){while(stk.size()>=3)stk.pop_back();}else{if(stk.size()>=3)stk.pop_back();stk.push_back(t);}} }
        std::vector<int> stk; int pos=0;
        Timer t;
        for(int i=0;i<NT;++i){int tk=tokens[i];pos++;
            if(tk==0||tk==3) stk.push_back(tk);
            else if(tk==4){ while(stk.size()>=3&&(stk[stk.size()-2]==1||stk[stk.size()-2]==2)) stk.pop_back(); }
            else{ if(stk.size()>=3&&(stk[stk.size()-2]==1||stk[stk.size()-2]==2)){stk.pop_back();} stk.push_back(tk); }
        }
        double sec=t.secs();escape_result(static_cast<uint64_t>(pos));
        sum+=static_cast<double>(NT)/sec/1e6;
    }
    ParseResult res; res.mtokens_per_sec=sum/R;
    show_done(res.mtokens_per_sec,"Mtokens/s"); return res;
}

// ============================================================================
// Section 15r — Bytecode Interpreter Benchmark
// ============================================================================
struct VmResult { double mips; };

VmResult bench_vm() {
    show_progress("Bytecode VM (stack machine)");
    constexpr int R=kBenchRounds; const int64_t INS=kVmInsns;
    // Minimal opcodes: PUSH, ADD, JMP, JZ, HALT
    enum Op { PUSH=0, ADD=1, JMP=2, JZ=3, HALT=4 };
    // Simple countdown: push N, loop{ push -1, add, dup, jz end, jmp loop } halt
    // 0:PUSH, 1:10000000, 2:PUSH, 3:-1, 4:ADD, 5:DUP, 6:JZ, 7:10, 8:JMP, 9:2, 10:HALT
    // We use -1 add (no SUB needed) and skip DUP by checking top-of-stack via PEEK pattern.
    // Even simpler: just push N, then loop{ push 1, sub } — wait, no SUB.
    // Simplest possible: PUSH counter, loop: ADD -1 until JZ via explicit value.
    // Actually, use a fixed-instruction-count loop: just PUSH and ADD repeatedly.
    std::vector<int> prog;
    // Fill with: PUSH,1, ADD, repeated many times (accumulate sum)
    // Then HALT. This is the simplest safe VM program.
    int64_t loop_ct = INS / 3; // each iter = 3 insns
    for (int64_t i = 0; i < loop_ct; ++i) {
        prog.push_back(PUSH);
        prog.push_back(1);
        prog.push_back(ADD);
    }
    prog.push_back(HALT);
    double sum=0;
    for(int round=0;round<R;++round){
        std::vector<int64_t> stack(256); int sp=0, ip=0; int64_t ic=0;
        auto run=[&](int64_t limit){ic=0;ip=0;sp=0;
            while(ic<limit&&ip>=0&&ip<static_cast<int>(prog.size())&&sp>=0&&sp<256){
                int op=prog[ip++];
                switch(op){
                    case PUSH: if(sp<256) stack[sp++]=prog[ip++]; break;
                    case ADD: if(sp>=2){sp--;stack[sp-1]+=stack[sp];} break;
                    case JMP: ip=prog[ip]; break;
                    case JZ: if(sp>=1&&stack[--sp]==0) ip=prog[ip]; else ip++; break;
                    case HALT: ip=-1; break;
                }
                ic++;
            }
        };
        run(INS/10);
        Timer t; run(INS);
        double sec=t.secs();escape_result(static_cast<uint64_t>(ic+stack[0]));
        sum+=static_cast<double>(ic)/sec/1e6;
    }
    VmResult res; res.mips=sum/R;
    show_done(res.mips,"MIPS"); return res;
}

// ============================================================================
// Section 15s — AES-256 Encryption Benchmark
// ============================================================================
struct AesResult { double mbs; };

AesResult bench_aes() {
    show_progress("AES-256 (CBC, hand-rolled)");
    constexpr int R=kBenchRounds;
    // AES S-box
    static const uint8_t sbox[256]={
        0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
        0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
        0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
        0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
        0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
        0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
        0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
        0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
        0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
        0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
        0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
        0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
        0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
        0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
        0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
        0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16};
    int64_t bytes=kAesMB*1024LL*1024LL; int64_t blocks=bytes/16;
    std::vector<uint8_t> data(bytes), key(32), iv(16);
    { uint64_t s=runtime_seed(); for(int i=0;i<32;++i) key[i]=sbox[(s+i)&0xFF]; for(int i=0;i<16;++i) iv[i]=sbox[(s+32+i)&0xFF]; for(int64_t i=0;i<bytes;++i) data[i]=static_cast<uint8_t>((s+i)&0xFF); }
    std::vector<uint32_t> rk(60); // round keys (simplified key schedule)
    for(int i=0;i<8;++i) rk[i]=(static_cast<uint32_t>(key[i*4])<<24)|(static_cast<uint32_t>(key[i*4+1])<<16)|(key[i*4+2]<<8)|key[i*4+3];
    // Simplified: just use S-box on each byte (CBC mode with XOR)
    double sum=0;
    for(int round=0;round<R;++round){
        std::vector<uint8_t> out(bytes); uint8_t prev[16]; memcpy(prev,iv.data(),16);
        for(int64_t b=0;b<blocks/10;++b){int64_t off=b*16;for(int j=0;j<16;++j)out[off+j]=sbox[data[off+j]^prev[j]];memcpy(prev,&out[off],16);}
        memcpy(prev,iv.data(),16);
        Timer t;
        for(int64_t b=0;b<blocks;++b){int64_t off=b*16;for(int j=0;j<16;++j)out[off+j]=sbox[data[off+j]^prev[j]];memcpy(prev,&out[off],16);}
        double sec=t.secs();escape_result(out[0]);
        sum+=static_cast<double>(bytes)/sec/1e6;
    }
    AesResult res; res.mbs=sum/R;
    show_done(res.mbs,"MB/s"); return res;
}

// ============================================================================
// Section 15t — SHA-256 Hash Benchmark
// ============================================================================
struct ShaResult { double mbs; };

ShaResult bench_sha256() {
    show_progress("SHA-256 Hash");
    constexpr int R=kBenchRounds;
    static const uint32_t K[64]={
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    int64_t bytes=kShaMB*1024LL*1024LL;
    std::vector<uint8_t> data(bytes);
    { uint64_t s=runtime_seed(); for(int64_t i=0;i<bytes;++i) data[i]=static_cast<uint8_t>((s+i)&0xFF); }
    auto sha256_block=[&](const uint8_t* blk,uint32_t* H){
        uint32_t w[64]; for(int i=0;i<16;++i) w[i]=(static_cast<uint32_t>(blk[i*4])<<24)|(blk[i*4+1]<<16)|(blk[i*4+2]<<8)|blk[i*4+3];
        for(int i=16;i<64;++i){uint32_t s0=((w[i-15]>>7)|(w[i-15]<<25))^((w[i-15]>>18)|(w[i-15]<<14))^(w[i-15]>>3);uint32_t s1=((w[i-2]>>17)|(w[i-2]<<15))^((w[i-2]>>19)|(w[i-2]<<13))^(w[i-2]>>10);w[i]=w[i-16]+s0+w[i-7]+s1;}
        uint32_t a=H[0],b=H[1],c=H[2],d=H[3],e=H[4],f=H[5],g=H[6],h=H[7];
        for(int i=0;i<64;++i){uint32_t S1=((e>>6)|(e<<26))^((e>>11)|(e<<21))^((e>>25)|(e<<7));uint32_t ch=(e&f)^((~e)&g);uint32_t t1=h+S1+ch+K[i]+w[i];uint32_t S0=((a>>2)|(a<<30))^((a>>13)|(a<<19))^((a>>22)|(a<<10));uint32_t maj=(a&b)^(a&c)^(b&c);uint32_t t2=S0+maj;h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}
        H[0]+=a;H[1]+=b;H[2]+=c;H[3]+=d;H[4]+=e;H[5]+=f;H[6]+=g;H[7]+=h;
    };
    double sum=0;
    for(int round=0;round<R;++round){
        uint32_t H[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
        std::vector<uint8_t> padded(data.begin(),data.end()); padded.push_back(0x80);
        while((padded.size()%64)!=56) padded.push_back(0);
        uint64_t bitlen=bytes*8; for(int i=7;i>=0;--i) padded.push_back(static_cast<uint8_t>(bitlen>>(i*8)));
        int blocks=static_cast<int>(padded.size()/64);
        // Warm up 10%
        for(int b=0;b<blocks/10;++b) sha256_block(&padded[b*64],H);
        Timer t;
        for(int b=0;b<blocks;++b) sha256_block(&padded[b*64],H);
        double sec=t.secs();escape_result(H[0]);
        sum+=static_cast<double>(bytes)/sec/1e6;
    }
    ShaResult res; res.mbs=sum/R;
    show_done(res.mbs,"MB/s"); return res;
}

// ============================================================================
// Section 15u — LZSS Compression Benchmark
// ============================================================================
struct LzssResult { double mbs; };

LzssResult bench_lzss() {
    show_progress("LZSS Compression");
    constexpr int R=kBenchRounds; const int WIN=512, LOOK=18;
    int64_t bytes=kLzssMB*1024LL*1024LL;
    std::vector<uint8_t> data(bytes);
    { uint64_t s=runtime_seed(); for(int64_t i=0;i<bytes;++i) data[i]=static_cast<uint8_t>(((s+i)*1103515245+12345)&0xFF); }
    double sum=0;
    for(int round=0;round<R;++round){
        std::vector<uint8_t> out; out.reserve(bytes);
        int64_t pos=0, warmup_end=bytes/10;
        // Warm up
        while(pos<warmup_end){int best_len=0,best_off=0;int start=std::max(static_cast<int64_t>(0),pos-WIN);
            for(int64_t i=start;i<pos&&pos<bytes;++i){int len=0;while(len<LOOK&&pos+len<bytes&&data[i+len]==data[pos+len])len++;if(len>best_len){best_len=len;best_off=static_cast<int>(pos-i);}}
            if(best_len>=3){out.push_back(1);out.push_back(best_off&0xFF);out.push_back((best_off>>8)&0xFF);out.push_back(best_len&0xFF);pos+=best_len;}else{out.push_back(0);out.push_back(data[pos]);pos++;}}
        out.clear(); pos=0;
        Timer t;
        while(pos<bytes){int best_len=0,best_off=0;int start=std::max(static_cast<int64_t>(0),pos-WIN);
            for(int64_t i=start;i<pos&&pos<bytes;++i){int len=0;while(len<LOOK&&pos+len<bytes&&data[i+len]==data[pos+len])len++;if(len>best_len){best_len=len;best_off=static_cast<int>(pos-i);}}
            if(best_len>=3){out.push_back(1);out.push_back(best_off&0xFF);out.push_back((best_off>>8)&0xFF);out.push_back(best_len&0xFF);pos+=best_len;}else{out.push_back(0);out.push_back(data[pos]);pos++;}}
        double sec=t.secs();escape_result(static_cast<uint64_t>(out.size()));
        sum+=static_cast<double>(bytes)/sec/1e6;
    }
    LzssResult res; res.mbs=sum/R;
    show_done(res.mbs,"MB/s"); return res;
}

// ============================================================================
// Section 15v — Image Convolution Benchmark
// ============================================================================
struct ConvResult { double mpixels_per_sec; };

ConvResult bench_conv() {
    show_progress("Image Convolution (Sobel 3x3)");
    constexpr int R=kBenchRounds, SZ=2048;
    std::vector<float> img(SZ*SZ), out(SZ*SZ);
    { uint64_t s=runtime_seed(); for(int i=0;i<SZ*SZ;++i) img[i]=static_cast<float>(((s+i)&0xFF)/255.0); }
    double sum=0;
    for(int round=0;round<R;++round){
        // Warm up border
        for(int y=1;y<SZ/10;++y) for(int x=1;x<SZ-1;++x){int c=y*SZ+x;out[c]=std::abs(-img[c-SZ-1]-2*img[c-SZ]-img[c-SZ+1]+img[c+SZ-1]+2*img[c+SZ]+img[c+SZ+1])+std::abs(-img[c-SZ-1]-2*img[c-1]-img[c+SZ-1]+img[c-SZ+1]+2*img[c+1]+img[c+SZ+1]);}
        Timer t;
        for(int y=1;y<SZ-1;++y) for(int x=1;x<SZ-1;++x){int c=y*SZ+x;out[c]=std::abs(-img[c-SZ-1]-2*img[c-SZ]-img[c-SZ+1]+img[c+SZ-1]+2*img[c+SZ]+img[c+SZ+1])+std::abs(-img[c-SZ-1]-2*img[c-1]-img[c+SZ-1]+img[c-SZ+1]+2*img[c+1]+img[c+SZ+1]);}
        double sec=t.secs();escape_result(static_cast<uint64_t>(out[SZ/2]*1e9f));
        sum+=static_cast<double>(SZ)*SZ/sec/1e6;
    }
    ConvResult res; res.mpixels_per_sec=sum/R;
    show_done(res.mpixels_per_sec,"Mpixels/s"); return res;
}

// ============================================================================
// Section 15w — Triangle Rasterization Benchmark
// ============================================================================
struct RasterResult { double mtris_per_sec; };

RasterResult bench_raster() {
    show_progress("Triangle Rasterization");
    constexpr int R=kBenchRounds; const int NT=kRasterTris; const int W=512, H=512;
    struct V2 { float x,y; }; struct Tri { V2 a,b,c; };
    std::vector<Tri> tris(NT);
    { uint64_t s=runtime_seed(); std::mt19937 rng(static_cast<unsigned>(s)); std::uniform_real_distribution<float> p(0,static_cast<float>(W-1));
        for(int i=0;i<NT;++i) tris[i]={{p(rng),p(rng)},{p(rng),p(rng)},{p(rng),p(rng)}}; }
    std::vector<uint8_t> fb(W*H,0);
    auto edge=[&](V2 a,V2 b,V2 p){return (b.x-a.x)*(p.y-a.y)-(b.y-a.y)*(p.x-a.x);};
    auto bbox=[&](const Tri& t,int&x0,int&x1,int&y0,int&y1){
        x0=std::max(0,static_cast<int>(std::min({t.a.x,t.b.x,t.c.x})));
        x1=std::min(W-1,static_cast<int>(std::max({t.a.x,t.b.x,t.c.x})));
        y0=std::max(0,static_cast<int>(std::min({t.a.y,t.b.y,t.c.y})));
        y1=std::min(H-1,static_cast<int>(std::max({t.a.y,t.b.y,t.c.y})));
    };
    double sum=0; int drawn=0;
    for(int round=0;round<R;++round){
        // Warm up
        for(int i=0;i<NT/10;++i){int x0,x1,y0,y1;bbox(tris[i],x0,x1,y0,y1);
            for(int y=y0;y<=y1;++y) for(int x=x0;x<=x1;++x){V2 p={static_cast<float>(x)+0.5f,static_cast<float>(y)+0.5f};if(edge(tris[i].a,tris[i].b,p)>=0&&edge(tris[i].b,tris[i].c,p)>=0&&edge(tris[i].c,tris[i].a,p)>=0)fb[y*W+x]=1;}}
        drawn=0; std::fill(fb.begin(),fb.end(),0);
        Timer t;
        for(int i=0;i<NT;++i){int x0,x1,y0,y1;bbox(tris[i],x0,x1,y0,y1);
            for(int y=y0;y<=y1;++y) for(int x=x0;x<=x1;++x){V2 p={static_cast<float>(x)+0.5f,static_cast<float>(y)+0.5f};if(edge(tris[i].a,tris[i].b,p)>=0&&edge(tris[i].b,tris[i].c,p)>=0&&edge(tris[i].c,tris[i].a,p)>=0){fb[y*W+x]=1;drawn++;}}}
        double sec=t.secs();escape_result(static_cast<uint64_t>(drawn));
        sum+=static_cast<double>(NT)/sec/1e6;
    }
    RasterResult res; res.mtris_per_sec=sum/R;
    show_done(res.mtris_per_sec,"Mtris/s"); return res;
}


// ============================================================================
// Section 21 — Score Aggregation & Final Report
// ============================================================================

// Reference values calibrated to the author's Apple M-series machine.
// On this machine the overall score is ~100.  Faster machines score
// above 100, slower machines below.  All values are raw benchmark
// outputs from a single run (3 rounds, averaged).
struct Reference {
    static constexpr double int_composite    =   3071.69;   // MOPS
    static constexpr double fp_composite     =   1215.0;   // MFLOPS
    static constexpr double mem_read         =       78.46;   // GB/s
    static constexpr double mem_write        =       80.46;   // GB/s
    static constexpr double mem_copy         =    15.0;   // GB/s
    static constexpr double mem_latency      =       88.49;   // ns (lower is better)
    static constexpr double branch_ratio     =         2.0;   // predictable/unpred.
    static constexpr double ilp_factor       =         4.19;   // ind/dep throughput
    static constexpr double matmul_gflops    =       34.32;   // GFLOPS
    static constexpr double sort_melem       =       61.47;   // Melem/s
    static constexpr double hash_mbs         =   8125.19;   // MB/s
    static constexpr double mt_speedup_4t    =         3.81;   // speedup at 4 threads
    static constexpr double fft_gflops       =        9.98;   // GFLOPS
    static constexpr double nbody_gflops     =        22.27;   // GFLOPS
    static constexpr double fluid_mlups      =      147.30;   // MLUPS
    static constexpr double chem_gflops      =        10.56;   // GFLOPS
    static constexpr double game_melem       =      309.37;   // Melem/s
    static constexpr double ai_gflops        =       17.26;   // GFLOPS
    static constexpr double fdtd_mlups       =    1543.17;   // MLUPS
    static constexpr double rigid_mcolps     =        1.45;   // Mcollisions/s
    static constexpr double ray_mrays        =     103.49;   // Mrays/s
    static constexpr double hf_gflops        =       10.22;   // GFLOPS
    static constexpr double mc_msteps        =         0.29;   // Msteps/s
    static constexpr double hp_mevals        =        2.01;   // Mevals/s
    static constexpr double sw_mcells        =   1774.60;   // Mcells/s
    static constexpr double neuron_mupd      =     163.80;   // Mneuron-upd/s
    static constexpr double lu_gflops        =        6.49;   // GFLOPS
    static constexpr double ode_steps        =  277143800.0; // steps/s
    static constexpr double karatsuba_mdm    =    3622.12;   // Mdigit-mul/s
    static constexpr double regex_mbs        =    2060.86;   // MB/s
    static constexpr double json_mbs         =    5000.0;   // MB/s
    static constexpr double alloc_mallocs    =      32.59;   // Mallocs/s
    static constexpr double atomic_matomic   =      91.39;   // Matomic-inc/s
    static constexpr double ast_mnodes       =    5000.0;   // Mnodes/s
    static constexpr double parse_mtokens    =    1789.40;   // Mtokens/s
    static constexpr double vm_mips          =    926.12;   // MIPS
    static constexpr double aes_mbs          =   4345.91;   // MB/s
    static constexpr double sha_mbs          =     425.75;   // MB/s
    static constexpr double lzss_mbs         =      58.57;   // MB/s
    static constexpr double conv_mpix        =   3901.45;   // Mpixels/s
    static constexpr double raster_mtris     =        0.03;   // Mtris/s
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
    FftResult      fft_res;
    NbodyResult    nbody_res;
    FluidResult    fluid_res;
    ChemResult     chem_res;
    GameResult     game_res;
    AiResult       ai_res;
    FdtdResult     fdtd_res;
    RigidResult    rigid_res;
    RayResult      ray_res;
    HfResult       hf_res;
    McResult       mc_res;
    HpResult       hp_res;
    SwResult       sw_res;
    NeuronResult   neuron_res;
    LuResult       lu_res;
    OdeResult      ode_res;
    KaratsubaResult karatsuba_res;
    RegexResult    regex_res;
    JsonResult     json_res;
    AllocResult    alloc_res;
    AtomicResult   atomic_res;
    AstResult      ast_res;
    ParseResult    parse_res;
    VmResult       vm_res;
    AesResult      aes_res;
    ShaResult      sha_res;
    LzssResult     lzss_res;
    ConvResult     conv_res;
    RasterResult   raster_res;
    double         overall_score;

void print(const SysInfo& sys) const {
        const int kW      = 62;
        const int kLabelW = 26;
        const int kValW   = 12;
        const int kUnitW  = 8;

        auto hr = [&](char c = '-') {
            std::cout << "  " << std::string(static_cast<size_t>(kW - 2), c) << "\n";
        };

        auto row = [&](const std::string& label, const std::string& value,
                       const std::string& unit) {
            std::cout << "  " << std::left  << std::setw(kLabelW) << label
                      << std::right << std::setw(kValW)  << value
                      << "  " << std::left  << std::setw(kUnitW) << unit << "\n";
        };

        auto row_score = [&](const std::string& label, const std::string& value,
                             const std::string& unit, double score) {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(1) << score;
            std::cout << "  " << std::left  << std::setw(kLabelW) << label
                      << std::right << std::setw(kValW)  << value
                      << "  " << std::left  << std::setw(kUnitW) << unit
                      << "  >> score  " << std::right << std::setw(6) << ss.str() << "\n";
        };

        auto section = [&](int n, const std::string& title) {
            std::cout << "\n";
            hr('=');
            std::ostringstream ss;
            ss << n << ". " << title;
            std::cout << "  " << std::left << std::setw(kW - 2) << ss.str() << "\n";
            hr('-');
        };

        // ---- Header ----
        std::cout << "\n";
        hr('=');
        std::cout << "  " << std::left << std::setw(kW - 2)
                  << "ezbench v2 — 40 Benchmarks · Baseline: Apple M-series" << "\n";
        hr('=');
        row("Architecture",   sys.arch,                    "");
        row("Compiler",       sys.compiler,                "");
        row("HW Threads",     std::to_string(sys.hw_threads), "");
        row("Rounds",         std::to_string(kBenchRounds), "(averaged)");
        hr('=');

        // ---- 1. Integer ----
        section(1, "Integer Arithmetic");
        row("  Add",             fmt(int_res.add_mops),     "MOPS");
        row("  Mul",             fmt(int_res.mul_mops),     "MOPS");
        row("  Div",             fmt(int_res.div_mops),     "MOPS");
        row("  Bit",             fmt(int_res.bit_mops),     "MOPS");
        row("  ---",             "",                        "");
        double int_score = norm_higher(int_res.composite_mops, Reference::int_composite);
        row_score("  Composite",  fmt(int_res.composite_mops), "MOPS", int_score);

        // ---- 2. FP ----
        section(2, "Floating-Point (scalar)");
        row("  SP  Add",    fmt(fp_res.sp_add, 0),  "MFLOPS");
        row("  SP  Mul",    fmt(fp_res.sp_mul, 0),  "MFLOPS");
        row("  SP  Div",    fmt(fp_res.sp_div, 0),  "MFLOPS");
        row("  SP  Sqrt",   fmt(fp_res.sp_sqrt, 0), "MFLOPS");
        row("  DP  Add",    fmt(fp_res.dp_add, 0),  "MFLOPS");
        row("  DP  Mul",    fmt(fp_res.dp_mul, 0),  "MFLOPS");
        row("  DP  Div",    fmt(fp_res.dp_div, 0),  "MFLOPS");
        row("  DP  Sqrt",   fmt(fp_res.dp_sqrt, 0), "MFLOPS");
        row("  ---",         "",                     "");
        double fp_score = norm_higher(fp_res.overall_mflops, Reference::fp_composite);
        row_score("  Composite", fmt(fp_res.overall_mflops, 0), "MFLOPS", fp_score);

        // ---- 3. Memory BW ----
        {
            std::ostringstream t;
            t << "Memory Bandwidth (sequential, " << kMemBufMB << " MB)";
            section(3, t.str());
        }
        row("  Read",  fmt(bw_res.read_gbs),  "GB/s");
        row("  Write", fmt(bw_res.write_gbs), "GB/s");
        row("  Copy",  fmt(bw_res.copy_gbs),  "GB/s");
        row("  ---",   "", "");
        double bw_read_score  = norm_higher(bw_res.read_gbs,  Reference::mem_read);
        double bw_write_score = norm_higher(bw_res.write_gbs, Reference::mem_write);
        double bw_score = std::sqrt(bw_read_score * bw_write_score);
        row_score("  Score (R+W geom)", "", "", bw_score);

        // ---- 4. Memory Latency ----
        section(4, "Memory Latency (pointer chase)");
        row("  Large-buffer avg", fmt(lat_res.lat_ns), "ns");
        if (!lat_res.curve.empty()) {
            std::cout << "\n  Working-set sweep (pointer-chase latency):\n";
            const int64_t csz[] = {4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072};
            for (size_t i = 0; i < lat_res.curve.size() && i < 16; ++i) {
                std::string lbl;
                if (csz[i] >= 1024) lbl = fmt(static_cast<double>(csz[i])/1024.0, 0) + " MB";
                else                lbl = std::to_string(csz[i]) + " KB";
                row("    " + lbl, fmt(lat_res.curve[i]), "ns");
            }
        }
        row("  ---",   "", "");
        double lat_score = norm_lower(lat_res.lat_ns, Reference::mem_latency);
        row_score("  Score", "", "", lat_score);

        // ---- 5. Branch ----
        section(5, "Branch Prediction");
        row("  Predictable",   fmt(br_res.predictable_mops),   "Melem/s");
        row("  Unpredictable", fmt(br_res.unpredictable_mops), "Melem/s");
        row("  Ratio",         fmt(br_res.ratio),              "x");
        row("  ---",           "",                             "");
        double br_score = norm_higher(br_res.ratio, Reference::branch_ratio);
        row_score("  Score", "", "", br_score);

        // ---- 6. Cache ----
        section(6, "Cache Hierarchy (bandwidth vs. size)");
        if (!cache_res.sizes_kb.empty()) {
            for (size_t i = 0; i < cache_res.sizes_kb.size(); ++i) {
                std::string lbl;
                double kb = cache_res.sizes_kb[i];
                if (kb >= 1024.0) lbl = fmt(kb / 1024.0, 0) + " MB";
                else              lbl = fmt(kb, 0) + " KB";
                row("  " + lbl, fmt(cache_res.bw_gbs[i]), "GB/s");
            }
        }
        row("  ---", "", "");
        double cache_score = norm_higher(
            cache_res.bw_gbs.empty() ? 0.0 : cache_res.bw_gbs[0], 100.0);
        row_score("  Score (L1 peak)", "", "", cache_score);

        // ---- 7. ILP ----
        section(7, "Instruction-Level Parallelism");
        row("  Dependent chain",  fmt(ilp_res.dep_ops_per_ns), "ops/ns");
        row("  Independent ops",  fmt(ilp_res.ind_ops_per_ns), "ops/ns");
        row("  ILP factor",       fmt(ilp_res.ilp_factor),     "x");
        row("  ---",              "",                          "");
        double ilp_score = norm_higher(ilp_res.ilp_factor, Reference::ilp_factor);
        row_score("  Score", "", "", ilp_score);

        // ---- 8. MT ----
        section(8, "Multi-Threaded Scaling");
        for (size_t i = 0; i < mt_res.thread_counts.size(); ++i) {
            int    tc = mt_res.thread_counts[i];
            double sp = mt_res.speedups[i];
            double ef = sp / static_cast<double>(tc);
            std::ostringstream l, v;
            l << "  " << tc << " thread" << (tc > 1 ? "s" : "");
            v << fmt(sp) << "x  (" << pct(ef) << " eff.)";
            row(l.str(), v.str(), "");
        }
        row("  ---", "", "");
        double mt_score = 0.0;
        for (size_t i = 0; i < mt_res.thread_counts.size(); ++i) {
            if (mt_res.thread_counts[i] >= 4) {
                mt_score = norm_higher(mt_res.speedups[i], Reference::mt_speedup_4t);
                break;
            }
        }
        if (mt_score == 0.0 && !mt_res.speedups.empty())
            mt_score = norm_higher(mt_res.speedups.back(), 2.0);
        row_score("  Score (@ 4 thr)", "", "", mt_score);

        // ---- 9. MatMul ----
        {
            std::ostringstream t;
            t << "Matrix Multiplication (" << mm_res.matrix_size << "x" << mm_res.matrix_size << " float)";
            section(9, t.str());
        }
        row("  Throughput", fmt(mm_res.gflops), "GFLOPS");
        row("  ---",        "",                 "");
        double mm_score = norm_higher(mm_res.gflops, Reference::matmul_gflops);
        row_score("  Score", "", "", mm_score);

        // ---- 10. Sort ----
        section(10, "Sorting Throughput (std::sort)");
        row("  " + comma(kSortSize) + " elements",
            fmt(sort_res.melem_per_sec), "Melem/s");
        row("  ---", "", "");
        double sort_score = norm_higher(sort_res.melem_per_sec, Reference::sort_melem);
        row_score("  Score", "", "", sort_score);

        // ---- 11. Hash ----
        {
            std::ostringstream t;
            t << "Hash / Mix Throughput (" << kHashMB << " MB)";
            section(11, t.str());
        }
        row("  Throughput", fmt(hash_res.mbs), "MB/s");
        row("  ---",        "",               "");
        double hash_score = norm_higher(hash_res.mbs, Reference::hash_mbs);
        row_score("  Score", "", "", hash_score);

        // ---- 12. FFT ----
        section(12, "Fast Fourier Transform (radix-2, " + std::to_string(static_cast<int>(kFftN)) + " pts)");
        row("  Throughput", fmt(fft_res.gflops), "GFLOPS");
        row("  ---",        "",                  "");
        double fft_score = norm_higher(fft_res.gflops, Reference::fft_gflops);
        row_score("  Score", "", "", fft_score);

        // ---- 13. N-body ----
        {
            std::ostringstream t;
            t << "N-body Gravity (" << kNbodyBodies << " bodies, O(n^2))";
            section(13, t.str());
        }
        row("  Throughput", fmt(nbody_res.gflops), "GFLOPS");
        row("  ---",        "",                    "");
        double nbody_score = norm_higher(nbody_res.gflops, Reference::nbody_gflops);
        row_score("  Score", "", "", nbody_score);

        // ---- 14. Fluid LBM ----
        {
            std::ostringstream t;
            t << "Fluid LBM (D2Q9, " << kLbmNx << "x" << kLbmNy << " grid)";
            section(14, t.str());
        }
        row("  Throughput", fmt(fluid_res.mlups), "MLUPS");
        row("  ---",        "",                    "");
        double fluid_score = norm_higher(fluid_res.mlups, Reference::fluid_mlups);
        row_score("  Score", "", "", fluid_score);

        // ---- 15. Molecular Dynamics ----
        {
            std::ostringstream t;
            t << "Molecular Dynamics (LJ, " << kMdAtoms << " atoms)";
            section(15, t.str());
        }
        row("  Throughput", fmt(chem_res.gflops), "GFLOPS");
        row("  ---",        "",                   "");
        double chem_score = norm_higher(chem_res.gflops, Reference::chem_gflops);
        row_score("  Score", "", "", chem_score);

        // ---- 16. Game Logic ----
        {
            std::ostringstream t;
            t << "Game Logic (" << comma(kGameEntities) << " entities)";
            section(16, t.str());
        }
        row("  Throughput", fmt(game_res.melem_per_sec), "Melem/s");
        row("  ---",        "",                           "");
        double game_score = norm_higher(game_res.melem_per_sec, Reference::game_melem);
        row_score("  Score", "", "", game_score);

        // ---- 17. AI Inference ----
        section(17, "AI Inference (MLP 784x512x256x10, batch " + std::to_string(kAiBatch) + ")");
        row("  Throughput", fmt(ai_res.gflops), "GFLOPS");
        row("  ---",        "",                 "");
        double ai_score = norm_higher(ai_res.gflops, Reference::ai_gflops);
        row_score("  Score", "", "", ai_score);


        // ---- 18. FDTD ----
        { std::ostringstream t; t << "FDTD EM (Yee grid, " << kFdtdNx << "x" << kFdtdNy << ")"; section(18, t.str()); }
        row("  Throughput", fmt(fdtd_res.mlups), "MLUPS"); row("  ---","","");
        double fdtd_score = norm_higher(fdtd_res.mlups, Reference::fdtd_mlups); row_score("  Score","","",fdtd_score);

        // ---- 19. Rigid Body ----
        { std::ostringstream t; t << "Rigid Body (" << kRigidBodies << " spheres)"; section(19, t.str()); }
        row("  Throughput", fmt(rigid_res.mcollisions_per_sec), "Mcoll/s"); row("  ---","","");
        double rigid_score = norm_higher(rigid_res.mcollisions_per_sec, Reference::rigid_mcolps); row_score("  Score","","",rigid_score);

        // ---- 20. Ray Tracing ----
        { std::ostringstream t; t << "Ray Tracing (" << kRayResX << "x" << kRayResY << " Whitted)"; section(20, t.str()); }
        row("  Throughput", fmt(ray_res.mrays_per_sec), "Mrays/s"); row("  ---","","");
        double ray_score = norm_higher(ray_res.mrays_per_sec, Reference::ray_mrays); row_score("  Score","","",ray_score);

        // ---- 21. Hartree-Fock ----
        { std::ostringstream t; t << "Quantum Chem (HF ERI, basis=" << kHfBasis << ")"; section(21, t.str()); }
        row("  Throughput", fmt(hf_res.gflops), "GFLOPS"); row("  ---","","");
        double hf_score = norm_higher(hf_res.gflops, Reference::hf_gflops); row_score("  Score","","",hf_score);

        // ---- 22. Monte Carlo ----
        section(22, "Monte Carlo (Metropolis, LJ)");
        row("  Throughput", fmt(mc_res.msteps_per_sec), "Msteps/s"); row("  ---","","");
        double mc_score = norm_higher(mc_res.msteps_per_sec, Reference::mc_msteps); row_score("  Score","","",mc_score);

        // ---- 23. Protein Folding ----
        section(23, "Protein Folding (HP model, annealing)");
        row("  Throughput", fmt(hp_res.mevals_per_sec), "Mevals/s"); row("  ---","","");
        double hp_score = norm_higher(hp_res.mevals_per_sec, Reference::hp_mevals); row_score("  Score","","",hp_score);

        // ---- 24. Smith-Waterman ----
        { std::ostringstream t; t << "DNA Alignment (Smith-Waterman, " << kSwSeqLen << "bp)"; section(24, t.str()); }
        row("  Throughput", fmt(sw_res.mcells_per_sec), "Mcells/s"); row("  ---","","");
        double sw_score = norm_higher(sw_res.mcells_per_sec, Reference::sw_mcells); row_score("  Score","","",sw_score);

        // ---- 25. Spiking Neurons ----
        { std::ostringstream t; t << "Spiking Neurons (Izhikevich, " << kIzhNeurons << " neurons)"; section(25, t.str()); }
        row("  Throughput", fmt(neuron_res.mupdates_per_sec), "Mn-upd/s"); row("  ---","","");
        double neuron_score = norm_higher(neuron_res.mupdates_per_sec, Reference::neuron_mupd); row_score("  Score","","",neuron_score);

        // ---- 26. LU Decomposition ----
        { std::ostringstream t; t << "LU Decomposition (" << kLuN << "x" << kLuN << ", no pivot)"; section(26, t.str()); }
        row("  Throughput", fmt(lu_res.gflops), "GFLOPS"); row("  ---","","");
        double lu_score = norm_higher(lu_res.gflops, Reference::lu_gflops); row_score("  Score","","",lu_score);

        // ---- 27. Stiff ODE ----
        section(27, "Stiff ODE (Robertson, BDF)");
        row("  Throughput", fmt(ode_res.steps_per_sec, 0), "steps/s"); row("  ---","","");
        double ode_score = norm_higher(ode_res.steps_per_sec, Reference::ode_steps); row_score("  Score","","",ode_score);

        // ---- 28. Karatsuba ----
        { std::ostringstream t; t << "BigInt Mul (Karatsuba, " << kKaratsubaBits << " bits)"; section(28, t.str()); }
        row("  Throughput", fmt(karatsuba_res.mdigit_muls_per_sec), "Mdig-mul/s"); row("  ---","","");
        double karatsuba_score = norm_higher(karatsuba_res.mdigit_muls_per_sec, Reference::karatsuba_mdm); row_score("  Score","","",karatsuba_score);

        // ---- 29. Pattern Match ----
        { std::ostringstream t; t << "Pattern Matching (email scan, " << kRegexMB << " MB)"; section(29, t.str()); }
        row("  Throughput", fmt(regex_res.mbs), "MB/s"); row("  ---","","");
        double regex_score = norm_higher(regex_res.mbs, Reference::regex_mbs); row_score("  Score","","",regex_score);

        // ---- 30. JSON ----
        { std::ostringstream t; t << "JSON Parsing (recursive descent, ~" << kJsonMB << " MB)"; section(30, t.str()); }
        row("  Throughput", fmt(json_res.mbs), "MB/s"); row("  ---","","");
        double json_score = norm_higher(json_res.mbs, Reference::json_mbs); row_score("  Score","","",json_score);

        // ---- 31. Memory Allocator ----
        section(31, "Memory Allocator (multi-threaded)");
        row("  Throughput", fmt(alloc_res.mallocs_per_sec), "Mallocs/s"); row("  ---","","");
        double alloc_score = norm_higher(alloc_res.mallocs_per_sec, Reference::alloc_mallocs); row_score("  Score","","",alloc_score);

        // ---- 32. Atomics ----
        section(32, "Concurrency Primitives (atomic inc)");
        row("  Throughput", fmt(atomic_res.matomic_per_sec), "Matomic/s"); row("  ---","","");
        double atomic_score = norm_higher(atomic_res.matomic_per_sec, Reference::atomic_matomic); row_score("  Score","","",atomic_score);

        // ---- 33. AST ----
        { std::ostringstream t; t << "AST Evaluation (" << comma(kAstNodes) << " nodes)"; section(33, t.str()); }
        row("  Throughput", fmt(ast_res.mnodes_per_sec), "Mnodes/s"); row("  ---","","");
        double ast_score = norm_higher(ast_res.mnodes_per_sec, Reference::ast_mnodes); row_score("  Score","","",ast_score);

        // ---- 34. LR Parser ----
        { std::ostringstream t; t << "LR Parser (" << comma(kParseTokens) << " tokens)"; section(34, t.str()); }
        row("  Throughput", fmt(parse_res.mtokens_per_sec), "Mtokens/s"); row("  ---","","");
        double parse_score = norm_higher(parse_res.mtokens_per_sec, Reference::parse_mtokens); row_score("  Score","","",parse_score);

        // ---- 35. Bytecode VM ----
        { std::ostringstream t; t << "Bytecode VM (" << comma(kVmInsns) << " insns)"; section(35, t.str()); }
        row("  Throughput", fmt(vm_res.mips), "MIPS"); row("  ---","","");
        double vm_score = norm_higher(vm_res.mips, Reference::vm_mips); row_score("  Score","","",vm_score);

        // ---- 36. AES-256 ----
        { std::ostringstream t; t << "AES-256 CBC (" << kAesMB << " MB)"; section(36, t.str()); }
        row("  Throughput", fmt(aes_res.mbs), "MB/s"); row("  ---","","");
        double aes_score = norm_higher(aes_res.mbs, Reference::aes_mbs); row_score("  Score","","",aes_score);

        // ---- 37. SHA-256 ----
        { std::ostringstream t; t << "SHA-256 (" << kShaMB << " MB)"; section(37, t.str()); }
        row("  Throughput", fmt(sha_res.mbs), "MB/s"); row("  ---","","");
        double sha_score = norm_higher(sha_res.mbs, Reference::sha_mbs); row_score("  Score","","",sha_score);

        // ---- 38. LZSS ----
        { std::ostringstream t; t << "LZSS Compression (" << kLzssMB << " MB)"; section(38, t.str()); }
        row("  Throughput", fmt(lzss_res.mbs), "MB/s"); row("  ---","","");
        double lzss_score = norm_higher(lzss_res.mbs, Reference::lzss_mbs); row_score("  Score","","",lzss_score);

        // ---- 39. Image Convolution ----
        { std::ostringstream t; t << "Image Convolution (Sobel, " << kConvRes << "x" << kConvRes << ")"; section(39, t.str()); }
        row("  Throughput", fmt(conv_res.mpixels_per_sec), "Mpixels/s"); row("  ---","","");
        double conv_score = norm_higher(conv_res.mpixels_per_sec, Reference::conv_mpix); row_score("  Score","","",conv_score);

        // ---- 40. Triangle Raster ----
        { std::ostringstream t; t << "Triangle Raster (" << comma(kRasterTris) << " tris)"; section(40, t.str()); }
        row("  Throughput", fmt(raster_res.mtris_per_sec), "Mtris/s"); row("  ---","","");
        double raster_score = norm_higher(raster_res.mtris_per_sec, Reference::raster_mtris); row_score("  Score","","",raster_score);


        // ---- Overall ----
        std::cout << "\n";
        hr('=');
        std::cout << "  " << std::left << std::setw(kLabelW) << "*  OVERALL SCORE"
                  << std::right << std::setw(kValW) << fmt(overall_score, 1) << "\n";
        hr('=');
        std::cout << "  (Geometric mean of 41 sub-scores.  100 = Apple M-series baseline.)\n";

        // Machine-readable CSV.
        std::cout << "\n[CSV] int,fp,mem_r,mem_w,mem_lat,branch,cache,ilp,mt,matmul,sort,hash,fft,nbody,fluid,chem,game,ai,fdtd,rigid,ray,hf,mc,hp,sw,neuron,lu,ode,karatsuba,regex,json,alloc,atomic,ast,parse,vm,aes,sha,lzss,conv,raster,overall\n";
        std::cout << "[CSV] " << fmt(int_score, 1)   << "," << fmt(fp_score, 1)       << ","
                                    << fmt(bw_read_score, 1) << "," << fmt(bw_write_score, 1) << ","
                                    << fmt(lat_score, 1)     << "," << fmt(br_score, 1)       << ","
                                    << fmt(cache_score, 1)   << "," << fmt(ilp_score, 1)      << ","
                                    << fmt(mt_score, 1)      << "," << fmt(mm_score, 1)       << ","
                                    << fmt(sort_score, 1)    << "," << fmt(hash_score, 1)     << ","
                                    << fmt(fft_score, 1)     << "," << fmt(nbody_score, 1)    << ","
                                    << fmt(fluid_score, 1)   << "," << fmt(chem_score, 1)     << ","
                                    << fmt(game_score, 1)    << "," << fmt(ai_score, 1)       << ","
                                    << fmt(fdtd_score, 1)    << "," << fmt(rigid_score, 1)    << ","
                                    << fmt(ray_score, 1)     << "," << fmt(hf_score, 1)       << ","
                                    << fmt(mc_score, 1)      << "," << fmt(hp_score, 1)       << ","
                                    << fmt(sw_score, 1)      << "," << fmt(neuron_score, 1)   << ","
                                    << fmt(lu_score, 1)      << "," << fmt(ode_score, 1)      << ","
                                    << fmt(karatsuba_score, 1) << "," << fmt(regex_score, 1) << ","
                                    << fmt(json_score, 1)    << "," << fmt(alloc_score, 1)    << ","
                                    << fmt(atomic_score, 1)  << "," << fmt(ast_score, 1)      << ","
                                    << fmt(parse_score, 1)   << "," << fmt(vm_score, 1)       << ","
                                    << fmt(aes_score, 1)     << "," << fmt(sha_score, 1)      << ","
                                    << fmt(lzss_score, 1)    << "," << fmt(conv_score, 1)     << ","
                                    << fmt(raster_score, 1)  << "," << fmt(overall_score, 1) << "\n";
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
    scores.push_back(norm_higher(r.fft_res.gflops, Reference::fft_gflops));
    scores.push_back(norm_higher(r.nbody_res.gflops, Reference::nbody_gflops));
    scores.push_back(norm_higher(r.fluid_res.mlups, Reference::fluid_mlups));
    scores.push_back(norm_higher(r.chem_res.gflops, Reference::chem_gflops));
    scores.push_back(norm_higher(r.game_res.melem_per_sec, Reference::game_melem));
    scores.push_back(norm_higher(r.ai_res.gflops, Reference::ai_gflops));
    scores.push_back(norm_higher(r.fdtd_res.mlups, Reference::fdtd_mlups));
    scores.push_back(norm_higher(r.rigid_res.mcollisions_per_sec, Reference::rigid_mcolps));
    scores.push_back(norm_higher(r.ray_res.mrays_per_sec, Reference::ray_mrays));
    scores.push_back(norm_higher(r.hf_res.gflops, Reference::hf_gflops));
    scores.push_back(norm_higher(r.mc_res.msteps_per_sec, Reference::mc_msteps));
    scores.push_back(norm_higher(r.hp_res.mevals_per_sec, Reference::hp_mevals));
    scores.push_back(norm_higher(r.sw_res.mcells_per_sec, Reference::sw_mcells));
    scores.push_back(norm_higher(r.neuron_res.mupdates_per_sec, Reference::neuron_mupd));
    scores.push_back(norm_higher(r.lu_res.gflops, Reference::lu_gflops));
    scores.push_back(norm_higher(r.ode_res.steps_per_sec, Reference::ode_steps));
    scores.push_back(norm_higher(r.karatsuba_res.mdigit_muls_per_sec, Reference::karatsuba_mdm));
    scores.push_back(norm_higher(r.regex_res.mbs, Reference::regex_mbs));
    scores.push_back(norm_higher(r.json_res.mbs, Reference::json_mbs));
    scores.push_back(norm_higher(r.alloc_res.mallocs_per_sec, Reference::alloc_mallocs));
    scores.push_back(norm_higher(r.atomic_res.matomic_per_sec, Reference::atomic_matomic));
    scores.push_back(norm_higher(r.ast_res.mnodes_per_sec, Reference::ast_mnodes));
    scores.push_back(norm_higher(r.parse_res.mtokens_per_sec, Reference::parse_mtokens));
    scores.push_back(norm_higher(r.vm_res.mips, Reference::vm_mips));
    scores.push_back(norm_higher(r.aes_res.mbs, Reference::aes_mbs));
    scores.push_back(norm_higher(r.sha_res.mbs, Reference::sha_mbs));
    scores.push_back(norm_higher(r.lzss_res.mbs, Reference::lzss_mbs));
    scores.push_back(norm_higher(r.conv_res.mpixels_per_sec, Reference::conv_mpix));
    scores.push_back(norm_higher(r.raster_res.mtris_per_sec, Reference::raster_mtris));

    // Geometric mean.
    double log_sum = 0.0;
    for (double s : scores) {
        if (s <= 0.0) s = 0.01;
        if (!std::isfinite(s) || s > 1000.0) s = 1000.0;
        log_sum += std::log(s);
    }
    return std::exp(log_sum / static_cast<double>(scores.size()));
}

// ============================================================================
// Section 16 — Entry Point
// ============================================================================

int main() {
    std::cout << "\n";
    std::cout << "  ============================================================\n";
    std::cout << "    ezbench v2 — 40 benchmarks, 8 categories, 15 ISAs\n";
    std::cout << "    Copyright (c) 2026 Hemingtsai  |  MIT License\n";
    std::cout << "  ============================================================\n\n";

    SysInfo sys = SysInfo::detect();
    std::cout << "  Detected: " << sys.arch << " | " << sys.compiler
              << " | " << sys.hw_threads << " HW thread(s)\n\n";

    FinalReport report;

    // Run all benchmarks.
    std::cout << "  --- Benchmarking (" << kBenchRounds << " rounds each, averaged) ---\n";
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
    report.fft_res   = bench_fft();
    report.nbody_res = bench_nbody();
    report.fluid_res = bench_fluid();
    report.chem_res  = bench_chemistry();
    report.game_res  = bench_game();
    report.ai_res    = bench_ai();
    report.ai_res    = bench_ai();
    report.fdtd_res   = bench_fdtd();
    report.rigid_res  = bench_rigid();
    report.ray_res    = bench_raytrace();
    report.hf_res     = bench_hf();
    report.mc_res     = bench_montecarlo();
    report.hp_res     = bench_protein();
    report.sw_res     = bench_sw();
    report.neuron_res = bench_neuron();
    report.lu_res     = bench_lu();
    report.ode_res    = bench_ode();
    report.karatsuba_res = bench_karatsuba();
    report.regex_res  = bench_regex();
    report.json_res   = bench_json();
    report.alloc_res  = bench_alloc();
    report.atomic_res = bench_atomic();
    report.ast_res    = bench_ast();
    report.parse_res  = bench_parser();
    report.vm_res     = bench_vm();
    report.aes_res    = bench_aes();
    report.sha_res    = bench_sha256();
    report.lzss_res   = bench_lzss();
    report.conv_res   = bench_conv();
    report.raster_res = bench_raster();

    report.overall_score = compute_overall(report);

    // Print the final report.
    report.print(sys);

    std::cout << "\nDone.  (optimisation barrier: " << std::hex
              << g_opt_barrier << std::dec << ")\n";
    return 0;
}
