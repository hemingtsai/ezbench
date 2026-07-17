# ezbench — Cross-Platform CPU Benchmark

A comprehensive, single-file CPU benchmark written in **ISO C++17**.  
No platform-specific headers, no intrinsics, no external dependencies — just the standard library.

Copyright &copy; 2026 **Hemingtsai** — Released under the [MIT License](LICENSE).

---

## Quick Start

```sh
# Build (any C++17 compiler)
g++ -std=c++17 -O2 -pthread -o ezbench ezbench.cpp

# Run
./ezbench
```

The entire benchmark runs in **30–120 seconds** depending on your CPU.

---

## What It Tests

| # | Benchmark | What It Measures |
|---|-----------|------------------|
| 1 | **Integer Arithmetic** | Throughput of add, multiply, divide, and bitwise/shift operations (MOPS) |
| 2 | **Floating-Point** | Scalar single- and double-precision add, mul, div, sqrt (MFLOPS) |
| 3 | **Memory Bandwidth** | Sequential read, write, and copy bandwidth on a large buffer (GB/s) |
| 4 | **Memory Latency** | Random-access pointer-chasing latency, plus a size sweep to reveal cache boundaries (ns) |
| 5 | **Branch Prediction** | Predictable vs. unpredictable branch throughput ratio |
| 6 | **Cache Hierarchy** | Read bandwidth at working-set sizes from 2 KB to 256 MB |
| 7 | **Instruction-Level Parallelism** | Dependent chain vs. independent operation throughput ratio |
| 8 | **Multi-Threaded Scaling** | Speedup and efficiency from 1 up to all hardware threads |
| 9 | **Matrix Multiplication** | Dense float matrix multiplication (GFLOPS) |
| 10 | **Sorting Throughput** | `std::sort` on random 64-bit integers (Melem/s) |
| 11 | **Hash / Mix Throughput** | FNV-1a style hash over a 128 MB buffer (MB/s) |

---

## Interpreting Results

Each sub-benchmark produces a **sub-score** normalised against a reference baseline
(a typical 6-core desktop CPU from ~2020). A score of **100** means “on par with the
reference”.

The **Overall Score** is the **geometric mean** of all sub-scores — a single number
that represents balanced, all-around CPU performance.

| Score Range | Interpretation |
|-------------|----------------|
| &lt; 60     | Entry-level / low-power |
| 60 – 90     | Mid-range |
| 90 – 120    | Upper mid-range (≈ reference) |
| 120 – 180   | High-end desktop / workstation |
| &gt; 180    | Top-tier / server-class |

A machine-readable CSV summary line is printed at the end for scripting:

```
[CSV] int,fp,mem_r,mem_w,mem_lat,branch,ilp,multithread,matmul,sort,hash,overall
[CSV] 69.1,63.1,470.5,620.1,86.6,56.3,104.8,113.6,128.7,543.5,295.3,153.1
```

---

## Design Principles

- **Zero platform-specific code** — uses only ISO C++17 and the standard library.
  Compiles and runs on Linux, macOS, Windows, and any other platform with a
  conforming C++17 toolchain.
- **Fair measurement** — runtime-dependent seeds prevent compile-time folding;
  `volatile` escape barriers prevent dead-code elimination; warm-up rounds
  stabilise cache and frequency state before timing.
- **Single file** — everything in `ezbench.cpp` with a navigable index at the top.
- **All comments in English.**

---

## File Structure

```
ezbench/
├── LICENSE          MIT License
├── README.md        This file
└── ezbench.cpp      The benchmark (single file, ~1360 lines)
```

---

## License

MIT — see [LICENSE](LICENSE) for full text.

---

## Author

**Hemingtsai** — 2026
