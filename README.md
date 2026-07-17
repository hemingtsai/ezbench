# ezbench — Cross-Platform CPU Benchmark

**40 benchmarks. 8 categories. One C++17 file. Zero dependencies.**

Runs on x86-64, AArch64, RISC-V, LoongArch, PowerPC, s390x, MIPS, ARM32 — any platform with a C++17 compiler.

Copyright &copy; 2026 **Hemingtsai** — [MIT License](LICENSE).

---

## Quick Start

```sh
g++ -std=c++17 -O2 -pthread -o ezbench ezbench.cpp
./ezbench
```

3 rounds averaged. ~30–120 seconds depending on CPU.

---

## Supported Architectures (15 ISAs)

x86-64, x86-32, AArch64, ARM32 (armhf / soft-float), RISC-V 64/32, LoongArch 64/32, PowerPC 64/32, IBM z/Architecture (s390x), IBM S/390, MIPS 64/32.

All detection is compile-time via predefined macros — zero runtime probing.

---

## Benchmark Categories (40 total)

### Compute
| # | Benchmark | Metric |
|---|-----------|--------|
| 1 | Integer Arithmetic (add/mul/div/bit) | MOPS |
| 2 | Floating-Point (SP/DP, add/mul/div/sqrt) | MFLOPS |
| 7 | Instruction-Level Parallelism | × factor |
| 9 | Matrix Multiplication (512² float) | GFLOPS |
| 12 | FFT (radix-2, 65536 pts) | GFLOPS |
| 26 | LU Decomposition | GFLOPS |
| 27 | Stiff ODE (Robertson) | steps/s |
| 28 | Karatsuba BigInt Multiplication | Mdigit-mul/s |

### Memory
| # | Benchmark | Metric |
|---|-----------|--------|
| 3 | Memory Bandwidth (read/write/copy) | GB/s |
| 4 | Memory Latency (pointer chase + cache sweep) | ns |
| 6 | Cache Hierarchy (2 KB → 256 MB) | GB/s |

### Physics Simulation
| # | Benchmark | Metric |
|---|-----------|--------|
| 13 | N-body Gravitational | GFLOPS |
| 18 | FDTD Electromagnetic (Yee grid) | MLUPS |
| 19 | Rigid Body Collision Detection | Mcoll/s |
| 20 | Whitted Ray Tracing | Mrays/s |

### Chemistry & Biology
| # | Benchmark | Metric |
|---|-----------|--------|
| 15 | Molecular Dynamics (Lennard-Jones) | GFLOPS |
| 21 | Quantum Chemistry (Hartree-Fock ERI) | GFLOPS |
| 22 | Monte Carlo (Metropolis) | Msteps/s |
| 23 | Protein Folding (HP model) | Mevals/s |
| 24 | DNA Alignment (Smith-Waterman) | Mcells/s |
| 25 | Spiking Neurons (Izhikevich) | Mn-upd/s |

### Fluids & Graphics
| # | Benchmark | Metric |
|---|-----------|--------|
| 14 | Fluid LBM (D2Q9) | MLUPS |
| 39 | Image Convolution (Sobel 3×3) | Mpixels/s |
| 40 | Triangle Rasterization | Mtris/s |

### Programming & Compilers
| # | Benchmark | Metric |
|---|-----------|--------|
| 16 | Game Logic (entity update) | Melem/s |
| 29 | Pattern Matching (email scan) | MB/s |
| 30 | JSON Parsing (recursive descent) | MB/s |
| 33 | AST Evaluation | Mnodes/s |
| 34 | LR Table-driven Parser | Mtokens/s |
| 35 | Bytecode Stack VM | MIPS |

### Concurrency
| # | Benchmark | Metric |
|---|-----------|--------|
| 8 | Multi-Threaded Scaling (1 → N threads) | × speedup |
| 31 | Memory Allocator (multi-threaded) | Mallocs/s |
| 32 | Concurrency Primitives (atomic inc) | Matomic/s |

### Crypto & Compression
| # | Benchmark | Metric |
|---|-----------|--------|
| 36 | AES-256 CBC (hand-rolled) | MB/s |
| 37 | SHA-256 | MB/s |
| 38 | LZSS Compression | MB/s |

### Misc
| # | Benchmark | Metric |
|---|-----------|--------|
| 5 | Branch Prediction | × ratio |
| 10 | Sorting (`std::sort`) | Melem/s |
| 11 | Hash / Mix (FNV-1a) | MB/s |
| 17 | AI Inference (MLP 784×512×256×10) | GFLOPS |

---

## Scoring

Each benchmark produces a **sub-score**: `raw / reference × 100`.

The **Overall Score** is the **geometric mean** of all 41 sub-scores.

**Baseline**: calibrated to the author's Apple M-series machine (score ≈ 100).
Faster CPUs score above 100; slower CPUs score below.

---

## Output

A clean column-aligned report with per-benchmark details, plus a machine-readable CSV line:

```
[CSV] int,fp,mem_r,mem_w,mem_lat,branch,...,overall
[CSV] 76.8,60.8,461.5,...,105.4
```

---

## Design

- **ISO C++17 only** — no intrinsics, no inline assembly, no platform headers
- **Fair** — runtime seeds prevent compile-time folding; `volatile` barriers prevent dead-code elimination; warm-up rounds stabilize caches; **3 rounds averaged**
- **Single file** — ~3100 lines with navigable index at top
- **All English comments**

---

## File Structure

```
ezbench/
├── LICENSE       MIT
├── README.md     This file
└── ezbench.cpp   The benchmark (~3100 lines)
```

---

## Author

**Hemingtsai** — 2026
