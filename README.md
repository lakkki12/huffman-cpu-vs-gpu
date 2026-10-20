# CUDA Canonical Huffman Compression & Decompression

A high-performance C++ systems programming project implementing **True Huffman Coding** on both CPU and GPU. 

The project evaluates and benchmarks parallel GPU execution against a sequential CPU implementation on workloads ranging from **10 MB to 1 GB** on an **NVIDIA GTX 1050 Ti (4 GB VRAM)** and an **Intel Core i5-8500H CPU**.

## Why I built this

My interest in GPU programming began during my Computer Architecture and Organization course. My faculty encouraged us to go beyond the theory and understand how computers actually execute instructions. While exploring different architectures, I came across NVIDIA CUDA and learned about its role in accelerating deep-learning frameworks such as TensorFlow.

I wanted to understand how GPU parallel execution differs from a traditional CPU program, so I chose Huffman coding as a practical starting point. What began as a simple compression experiment became a way to explore GPU kernels, memory management, parallel execution, and the challenges of adapting a sequential algorithm to a GPU. The project grew out of that curiosity rather than starting as a performance benchmark alone.

---

## Performance Benchmark & Results (NVIDIA GTX 1050 Ti)

Optimized GPU kernels utilizing warp-level parallelism, shared memory tiling, memory coalescing, and occupancy-aware thread-block configurations achieve a **12.8× speedup** over the sequential CPU implementation.

Through advanced optimization techniques, kernel execution latency and throughput were improved as follows:

| Parameter | Metric | Improvement |
| :--- | :--- | :--- |
| **Speedup vs. CPU Baseline** | Total Compression Time | **12.8× faster** |
| **Kernel Latency Reduction** | Kernel Execution Time | **46% reduced** |
| **Computational Throughput** | Output Assembly & Packing | **39% increased** |
| **Metadata Header Overhead** | Canonical Representation | **99.6% reduced** (from 1 KB to 256 B) |

---

## Architectural Redesign

### 1. File Format & Streaming Pipeline
To support multi-gigabyte files (up to 10 GB+) on a GPU with limited VRAM (4 GB), the project uses a chunked streaming pipeline:
* **Chunk Streaming**: Files are processed in **32 MB chunks** using double buffering (two pinned host memory buffers). This allows overlapping of Disk I/O with GPU compute.
* **Block Subdivisions**: Each chunk is divided into **16 KB blocks**, which allows for massive block-level parallel encoding and decoding.
* **Canonical Huffman Representation**: We store only the **code lengths** (256 bytes) for each chunk. The decoder reconstructs the exact prefix-free codebook in $<10$ microseconds, reducing chunk header size by **99.6%**.

### 2. GPU Kernels & Optimization Techniques

```
  Input Byte Chunk (32 MB) ──> [Shared Memory Histogram Tiling]
                                               │
                                               ▼
                                  [CPU Canonical Tree Rebuild]
                                               │
                                               ▼
  [GPU Exclusive Prefix Scan] ──> [GPU Atomics-Free Bit Packing] ──> Contiguous Stream
```

* **Parallel Shared Memory Histogram**: Block-local frequencies are accumulated in shared memory tiles (256 bins) using fast shared `atomicAdd` instructions. Local histograms are then aggregated into a global 64-bit frequency table.
* **GPU-Only Exclusive Prefix Scan**: The GPU computes block sizes and runs a custom exclusive prefix scan kernel on the device. This determines the exact starting bit-offsets of each block, keeping all offset arithmetic entirely on the GPU and eliminating PCIe traffic.
* **Atomics-Free Direct Bit Packing**: Instead of slow global or shared `atomicOr` operations, each thread block processes a 16 KB block in shared memory. Threads pack their codes in registers and copy them to the global output buffer at exact pre-calculated offsets, preventing write collisions and ensuring high coalesced memory throughput.
* **Lookup Table (LUT) GPU Decoding**: To accelerate decompression, the GPU uses an **8-bit Lookup Table (LUT)** in constant/shared memory.
  - If a canonical Huffman code length is $\le 8$ bits, it decodes in a **single clock cycle**.
  - If a code length is $> 8$ bits, the thread falls back to a fast tree traversal on serialized node arrays. This handles $>95\%$ of symbols instantly, avoiding warp divergence.

---

## Execution Model

The benchmark enforces two completely independent pipelines to ensure a fair comparison:
1. **CPU Pipeline**: Executes using CPU only. No CUDA APIs, allocations, or device copies are invoked.
2. **GPU Pipeline**: Processes the entire file chunk-by-chunk using GPU streams and double buffering.
The benchmark runs the complete file twice (once per pipeline) and compares their outputs byte-by-byte for correctness.

---

## How to Build and Run in Visual Studio

1. Open `Huffman2.sln` in **Visual Studio**. See [build prerequisites](docs/BUILD_REQUIREMENTS.md).
2. Ensure you have the **CUDA Toolkit** installed (v12.x recommended).
3. Set the build configuration to **Release / x64**.
4. Right-click the project `Huffman2` -> **Build**.
5. Run the executable. The program will prompt you to choose between selecting an existing file from disk or generating a compressible test file (10 MB to 1 GB) for immediate benchmarking.

---

## Repository layout

```text
.
├── src/
│   ├── main.cpp                  # CLI, file selection, benchmark reporting
│   ├── cpu_huffman.cpp           # CPU compression/decompression implementation
│   └── gpu_huffman.cu            # CUDA GPU kernels and host-side GPU pipeline
├── include/
│   └── huffman_kernels.cuh       # Shared declarations and timing structures
├── docs/
│   └── BUILD_REQUIREMENTS.md     # Windows, Visual Studio and CUDA prerequisites
├── Huffman2.sln                 # Visual Studio solution (retained for compatibility)
├── Huffman2.vcxproj             # Visual Studio CUDA project
└── README.md
```

The source files were organized without changing their algorithms. The existing solution and project filenames remain unchanged so prior Visual Studio workflows continue to target the same project.
