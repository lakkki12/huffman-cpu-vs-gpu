#include "kernels.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cstring>

// CUDA Error Checking Macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
                      << ": " << cudaGetErrorString(err) << "\n"; \
            exit(1); \
        } \
    } while (0)

// Helper structures for Canonical Huffman Tree generation on CPU
struct GpuTreeCpuNode {
    uint8_t symbol;
    uint64_t freq;
    GpuTreeCpuNode* left;
    GpuTreeCpuNode* right;

    GpuTreeCpuNode(uint8_t sym, uint64_t f) : symbol(sym), freq(f), left(nullptr), right(nullptr) {}
    ~GpuTreeCpuNode() {
        delete left;
        delete right;
    }
};

struct CompareGpuTreeCpuNode {
    bool operator()(GpuTreeCpuNode* l, GpuTreeCpuNode* r) {
        return l->freq > r->freq;
    }
};

static void getLengthsCpu(GpuTreeCpuNode* root, uint32_t depth, uint32_t* lengths) {
    if (!root) return;
    if (!root->left && !root->right) {
        lengths[root->symbol] = (depth == 0) ? 1 : depth;
        return;
    }
    getLengthsCpu(root->left, depth + 1, lengths);
    getLengthsCpu(root->right, depth + 1, lengths);
}

static void generateCanonicalCodesCpu(const uint32_t* lengths, HuffmanCode* codes) {
    struct SymbolLength {
        uint8_t symbol;
        uint32_t length;
    };

    std::vector<SymbolLength> activeSymbols;
    for (int i = 0; i < ALPHABET_SIZE; ++i) {
        if (lengths[i] > 0) {
            activeSymbols.push_back({ (uint8_t)i, lengths[i] });
            codes[i] = { 0, 0 };
        } else {
            codes[i] = { 0, 0 };
        }
    }

    std::sort(activeSymbols.begin(), activeSymbols.end(), [](const SymbolLength& a, const SymbolLength& b) {
        if (a.length != b.length) return a.length < b.length;
        return a.symbol < b.symbol;
    });

    uint64_t code = 0;
    uint32_t prevLen = 0;
    for (const auto& sym : activeSymbols) {
        if (prevLen > 0) {
            code <<= (sym.length - prevLen);
        }
        codes[sym.symbol] = { code, sym.length };
        code++;
        prevLen = sym.length;
    }
}

// Rebuild and serialize Canonical Tree for the GPU Decoder
static void buildDecoderTables(const uint32_t* lengths, DecEntry* lut, std::vector<int16_t>& left, std::vector<int16_t>& right, std::vector<uint8_t>& symbol) {
    HuffmanCode codes[ALPHABET_SIZE];
    generateCanonicalCodesCpu(lengths, codes);

    std::memset(lut, 0, sizeof(DecEntry) * 256);
    for (int i = 0; i < ALPHABET_SIZE; ++i) {
        if (lengths[i] > 0 && lengths[i] <= 8) {
            uint64_t code = codes[i].code;
            uint32_t len = codes[i].length;
            int shift = 8 - len;
            int start = (code << shift);
            int end = start + (1 << shift);
            for (int idx = start; idx < end; ++idx) {
                lut[idx].symbol = (uint8_t)i;
                lut[idx].len = (uint8_t)len;
            }
        }
    }

    left.clear();
    right.clear();
    symbol.clear();

    left.push_back(-1);
    right.push_back(-1);
    symbol.push_back(0);

    for (int i = 0; i < ALPHABET_SIZE; ++i) {
        if (lengths[i] > 0) {
            uint64_t code = codes[i].code;
            uint32_t len = codes[i].length;

            int16_t curr = 0;
            for (int bitIdx = (int)len - 1; bitIdx >= 0; --bitIdx) {
                uint32_t bit = (code >> bitIdx) & 1;
                if (bit == 0) {
                    if (left[curr] == -1) {
                        left[curr] = left.size();
                        left.push_back(-1);
                        right.push_back(-1);
                        symbol.push_back(0);
                    }
                    curr = left[curr];
                } else {
                    if (right[curr] == -1) {
                        right[curr] = right.size();
                        left.push_back(-1);
                        right.push_back(-1);
                        symbol.push_back(0);
                    }
                    curr = right[curr];
                }
            }
            symbol[curr] = (uint8_t)i;
        }
    }
}

// --- CUDA Kernels ---

// 1. Parallel Histogram Kernel (Shared Memory Tiling)
__global__ void gpuHistogramKernel(const uint8_t* d_input, uint32_t size, uint64_t* d_freq) {
    __shared__ uint32_t s_hist[ALPHABET_SIZE];
    int tid = threadIdx.x;
    s_hist[tid] = 0;
    __syncthreads();

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (uint32_t i = idx; i < size; i += stride) {
        atomicAdd(&s_hist[d_input[i]], 1);
    }
    __syncthreads();

    atomicAdd(&d_freq[tid], s_hist[tid]);
}

// 2. Parallel Compute Block Sizes (Dry-Run Bit Count)
__global__ void gpuComputeBlockSizesKernel(
    const uint8_t* d_input, uint32_t chunk_size,
    const HuffmanCode* d_codes,
    uint16_t* d_blockSizes
) {
    uint32_t blockIdxX = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t blockOffset = blockIdxX * BLOCK_SIZE;

    __shared__ uint32_t s_threadSizes[256];

    if (blockOffset >= chunk_size) {
        if (tid == 0) d_blockSizes[blockIdxX] = 0;
        return;
    }

    uint32_t currentBlockSize = (blockOffset + BLOCK_SIZE > chunk_size) ? (chunk_size - blockOffset) : BLOCK_SIZE;
    uint32_t threadStart = blockOffset + tid * 64;
    uint32_t bitCount = 0;

    for (uint32_t i = 0; i < 64; ++i) {
        uint32_t idx = threadStart + i;
        if (idx < chunk_size && idx < blockOffset + currentBlockSize) {
            uint8_t sym = d_input[idx];
            bitCount += d_codes[sym].length;
        }
    }

    uint32_t byteCount = (bitCount + 7) / 8;
    s_threadSizes[tid] = byteCount;
    __syncthreads();

    if (tid == 0) {
        uint32_t total = 0;
        for (int i = 0; i < 256; ++i) {
            total += s_threadSizes[i];
        }
        d_blockSizes[blockIdxX] = (uint16_t)total;
    }
}

// 3. GPU Exclusive Prefix Scan of Block Sizes
__global__ void gpuScanBlockSizesKernel(const uint16_t* d_sizes, uint32_t* d_offsets, uint32_t* d_totalSize, uint32_t numBlocks) {
    if (threadIdx.x == 0) {
        uint32_t sum = 0;
        for (uint32_t i = 0; i < numBlocks; ++i) {
            d_offsets[i] = sum;
            sum += d_sizes[i];
        }
        *d_totalSize = sum;
    }
}

// 4. Atomics-Free Bit Packing Kernel
__global__ void gpuEncodeBlocksKernel(
    const uint8_t* d_input, uint32_t chunk_size,
    const HuffmanCode* d_codes,
    const uint32_t* d_blockOffsets,
    uint8_t* d_output
) {
    uint32_t blockIdxX = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t blockOffset = blockIdxX * BLOCK_SIZE;

    if (blockOffset >= chunk_size) return;

    uint32_t currentBlockSize = (blockOffset + BLOCK_SIZE > chunk_size) ? (chunk_size - blockOffset) : BLOCK_SIZE;

    __shared__ uint32_t s_threadSizes[256];
    __shared__ uint32_t s_threadOffsets[256];
    __shared__ uint8_t s_temp[256][128]; // Shared memory tiling buffer

    uint32_t threadStart = blockOffset + tid * 64;

    uint8_t* threadBuf = &s_temp[tid][0];
    uint32_t bitCount = 0;
    uint32_t byteCount = 0;
    uint8_t currentByte = 0;

    for (uint32_t i = 0; i < 64; ++i) {
        uint32_t idx = threadStart + i;
        if (idx < chunk_size && idx < blockOffset + currentBlockSize) {
            uint8_t sym = d_input[idx];
            HuffmanCode c = d_codes[sym];

            for (int b = (int)c.length - 1; b >= 0; --b) {
                uint8_t bit = (c.code >> b) & 1;
                currentByte = (currentByte << 1) | bit;
                bitCount++;
                if (bitCount == 8) {
                    threadBuf[byteCount++] = currentByte;
                    currentByte = 0;
                    bitCount = 0;
                }
            }
        }
    }
    if (bitCount > 0) {
        currentByte <<= (8 - bitCount);
        threadBuf[byteCount++] = currentByte;
    }

    s_threadSizes[tid] = byteCount;
    __syncthreads();

    if (tid == 0) {
        uint32_t sum = 0;
        for (int i = 0; i < 256; ++i) {
            s_threadOffsets[i] = sum;
            sum += s_threadSizes[i];
        }
    }
    __syncthreads();

    uint32_t blockStartOffset = d_blockOffsets[blockIdxX];
    uint32_t myOffset = s_threadOffsets[tid];
    uint32_t mySize = s_threadSizes[tid];

    for (uint32_t i = 0; i < mySize; ++i) {
        d_output[blockStartOffset + myOffset + i] = threadBuf[i];
    }
}

// 5. GPU Bit Reader for Parallel Decoder
struct GpuBitReader {
    const uint8_t* buffer;
    uint32_t bit_buf = 0;
    int bit_count = 0;
    uint32_t byte_idx = 0;
    uint32_t max_bytes = 0;

    __device__ GpuBitReader(const uint8_t* buf, uint32_t size) : buffer(buf), max_bytes(size) {}

    __device__ void fill() {
        while (bit_count <= 24 && byte_idx < max_bytes) {
            bit_buf = (bit_buf << 8) | buffer[byte_idx++];
            bit_count += 8;
        }
    }

    __device__ uint32_t readBit() {
        if (bit_count == 0) {
            fill();
            if (bit_count == 0) return 0;
        }
        uint32_t bit = (bit_buf >> (bit_count - 1)) & 1;
        bit_count--;
        return bit;
    }

    __device__ uint32_t peek8() {
        if (bit_count < 8) {
            fill();
        }
        if (bit_count == 0) return 0;
        if (bit_count < 8) {
            return (bit_buf & ((1 << bit_count) - 1)) << (8 - bit_count);
        }
        return (bit_buf >> (bit_count - 8)) & 0xFF;
    }

    __device__ void consume(int count) {
        bit_count -= count;
    }
};

// 6. LUT-Based Parallel Decoding Kernel
__global__ void gpuDecodeBlocksKernel(
    const uint8_t* d_compressed, const uint32_t* d_blockOffsets, const uint16_t* d_blockSizes,
    uint8_t* d_output, size_t chunk_size,
    const DecEntry* d_lut, const int16_t* d_left, const int16_t* d_right, const uint8_t* d_symbol,
    uint32_t numBlocks
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBlocks) return;

    uint32_t blockOutOffset = idx * BLOCK_SIZE;
    uint32_t currentBlockSize = (idx == numBlocks - 1) ? (chunk_size - blockOutOffset) : BLOCK_SIZE;
    uint16_t compressedBlockSize = d_blockSizes[idx];

    GpuBitReader reader(d_compressed + d_blockOffsets[idx], compressedBlockSize);
    uint8_t* blockOut = d_output + blockOutOffset;

    for (uint32_t i = 0; i < currentBlockSize; ++i) {
        uint32_t val8 = reader.peek8();
        DecEntry entry = d_lut[val8];
        if (entry.len > 0) {
            blockOut[i] = entry.symbol;
            reader.consume(entry.len);
        } else {
            // Fallback to bit-by-bit tree traversal
            int16_t curr = 0;
            while (true) {
                uint32_t bit = reader.readBit();
                curr = (bit == 0) ? d_left[curr] : d_right[curr];
                if (d_left[curr] == -1) {
                    blockOut[i] = d_symbol[curr];
                    break;
                }
            }
        }
    }
}

// --- Host Routines ---

// GPU Compression Streaming Pipeline
void gpuCompress(const std::string& inputFilePath, const std::string& outputFilePath, GpuTimings& timings) {
    std::ifstream inFile(inputFilePath, std::ios::binary);
    std::ofstream outFile(outputFilePath, std::ios::binary);

    if (!inFile || !outFile) {
        std::cerr << "[GPU] Error opening files for compression.\n";
        return;
    }

    // Write file header
    const char magic[4] = { 'H', 'U', 'F', 'F' };
    outFile.write(magic, 4);

    inFile.seekg(0, std::ios::end);
    uint64_t originalFileSize = inFile.tellg();
    inFile.seekg(0, std::ios::beg);

    outFile.write(reinterpret_cast<const char*>(&originalFileSize), sizeof(originalFileSize));

    const size_t CHUNK_SIZE = 32 * 1024 * 1024; // 32 MB chunk

    // Pinned host memory buffers for double buffering
    uint8_t* h_inBuf[2];
    uint8_t* h_outBuf[2];
    uint16_t* h_blockSizes[2];

    CUDA_CHECK(cudaMallocHost(&h_inBuf[0], CHUNK_SIZE));
    CUDA_CHECK(cudaMallocHost(&h_inBuf[1], CHUNK_SIZE));
    CUDA_CHECK(cudaMallocHost(&h_outBuf[0], CHUNK_SIZE));
    CUDA_CHECK(cudaMallocHost(&h_outBuf[1], CHUNK_SIZE));

    size_t maxBlocks = (CHUNK_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CUDA_CHECK(cudaMallocHost(&h_blockSizes[0], maxBlocks * sizeof(uint16_t)));
    CUDA_CHECK(cudaMallocHost(&h_blockSizes[1], maxBlocks * sizeof(uint16_t)));

    // Device memory buffers
    uint8_t* d_inBuf;
    uint8_t* d_outBuf;
    uint64_t* d_freq;
    HuffmanCode* d_codes;
    uint16_t* d_blockSizes;
    uint32_t* d_blockOffsets;
    uint32_t* d_totalCompressedSize;

    CUDA_CHECK(cudaMalloc(&d_inBuf, CHUNK_SIZE));
    CUDA_CHECK(cudaMalloc(&d_outBuf, CHUNK_SIZE));
    CUDA_CHECK(cudaMalloc(&d_freq, ALPHABET_SIZE * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_codes, ALPHABET_SIZE * sizeof(HuffmanCode)));
    CUDA_CHECK(cudaMalloc(&d_blockSizes, maxBlocks * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_blockOffsets, maxBlocks * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_totalCompressedSize, sizeof(uint32_t)));

    // CUDA Streams
    cudaStream_t computeStream;
    CUDA_CHECK(cudaStreamCreate(&computeStream));

    // CUDA Events for timing
    cudaEvent_t h2dStart, h2dEnd;
    cudaEvent_t histStart, histEnd;
    cudaEvent_t scanStart, scanEnd;
    cudaEvent_t encodeStart, encodeEnd;
    cudaEvent_t d2hStart, d2hEnd;

    CUDA_CHECK(cudaEventCreate(&h2dStart));
    CUDA_CHECK(cudaEventCreate(&h2dEnd));
    CUDA_CHECK(cudaEventCreate(&histStart));
    CUDA_CHECK(cudaEventCreate(&histEnd));
    CUDA_CHECK(cudaEventCreate(&scanStart));
    CUDA_CHECK(cudaEventCreate(&scanEnd));
    CUDA_CHECK(cudaEventCreate(&encodeStart));
    CUDA_CHECK(cudaEventCreate(&encodeEnd));
    CUDA_CHECK(cudaEventCreate(&d2hStart));
    CUDA_CHECK(cudaEventCreate(&d2hEnd));

    uint64_t bytesProcessed = 0;
    int bufIdx = 0;

    // Prefetch first chunk disk read
    auto ioStart = std::chrono::high_resolution_clock::now();
    size_t currentChunkSize = std::min(originalFileSize - bytesProcessed, (uint64_t)CHUNK_SIZE);
    inFile.read(reinterpret_cast<char*>(h_inBuf[bufIdx]), currentChunkSize);
    auto ioEnd = std::chrono::high_resolution_clock::now();
    timings.diskReadTime += std::chrono::duration<double, std::milli>(ioEnd - ioStart).count();

    while (currentChunkSize > 0) {
        size_t nextChunkSize = 0;
        int nextBufIdx = 1 - bufIdx;

        if (bytesProcessed + currentChunkSize < originalFileSize) {
            nextChunkSize = std::min(originalFileSize - (bytesProcessed + currentChunkSize), (uint64_t)CHUNK_SIZE);
            ioStart = std::chrono::high_resolution_clock::now();
            inFile.read(reinterpret_cast<char*>(h_inBuf[nextBufIdx]), nextChunkSize);
            ioEnd = std::chrono::high_resolution_clock::now();
            timings.diskReadTime += std::chrono::duration<double, std::milli>(ioEnd - ioStart).count();
        }

        // 1. Host-to-Device Copy
        CUDA_CHECK(cudaEventRecord(h2dStart, computeStream));
        CUDA_CHECK(cudaMemcpyAsync(d_inBuf, h_inBuf[bufIdx], currentChunkSize, cudaMemcpyHostToDevice, computeStream));
        CUDA_CHECK(cudaEventRecord(h2dEnd, computeStream));

        // 2. Frequency counting (Histogram)
        CUDA_CHECK(cudaEventRecord(histStart, computeStream));
        CUDA_CHECK(cudaMemsetAsync(d_freq, 0, ALPHABET_SIZE * sizeof(uint64_t), computeStream));
        int histBlockSize = 256;
        int histGridSize = (currentChunkSize + histBlockSize - 1) / histBlockSize;
        histGridSize = std::min(histGridSize, 128);
        gpuHistogramKernel<<<histGridSize, histBlockSize, 0, computeStream>>>(d_inBuf, (uint32_t)currentChunkSize, d_freq);
        CUDA_CHECK(cudaEventRecord(histEnd, computeStream));

        // 3. Copy frequencies back to CPU
        uint64_t h_freq[ALPHABET_SIZE];
        CUDA_CHECK(cudaMemcpyAsync(h_freq, d_freq, ALPHABET_SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost, computeStream));
        CUDA_CHECK(cudaStreamSynchronize(computeStream));

        // 4. Tree building & Canonical Codebook generation on CPU
        auto cpuTreeStart = std::chrono::high_resolution_clock::now();
        std::priority_queue<GpuTreeCpuNode*, std::vector<GpuTreeCpuNode*>, CompareGpuTreeCpuNode> minHeap;
        for (int i = 0; i < ALPHABET_SIZE; ++i) {
            if (h_freq[i] > 0) {
                minHeap.push(new GpuTreeCpuNode((uint8_t)i, h_freq[i]));
            }
        }

        uint32_t lengths[ALPHABET_SIZE] = { 0 };
        if (!minHeap.empty()) {
            GpuTreeCpuNode* root = nullptr;
            if (minHeap.size() == 1) {
                GpuTreeCpuNode* single = minHeap.top(); minHeap.pop();
                root = new GpuTreeCpuNode(0, single->freq);
                root->left = single;
            } else {
                while (minHeap.size() > 1) {
                    GpuTreeCpuNode* left = minHeap.top(); minHeap.pop();
                    GpuTreeCpuNode* right = minHeap.top(); minHeap.pop();
                    GpuTreeCpuNode* parent = new GpuTreeCpuNode(0, left->freq + right->freq);
                    parent->left = left;
                    parent->right = right;
                    minHeap.push(parent);
                }
                root = minHeap.top(); minHeap.pop();
            }
            getLengthsCpu(root, 0, lengths);
            delete root;
        }

        HuffmanCode h_codes[ALPHABET_SIZE];
        generateCanonicalCodesCpu(lengths, h_codes);
        auto cpuTreeEnd = std::chrono::high_resolution_clock::now();
        timings.treeTime += std::chrono::duration<double, std::milli>(cpuTreeEnd - cpuTreeStart).count();

        // 5. Copy Canonical code table back to GPU
        CUDA_CHECK(cudaMemcpyAsync(d_codes, h_codes, ALPHABET_SIZE * sizeof(HuffmanCode), cudaMemcpyHostToDevice, computeStream));

        // 6. Launch Compute Block Sizes Kernel
        size_t numBlocks = (currentChunkSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        gpuComputeBlockSizesKernel<<<numBlocks, 256, 0, computeStream>>>(d_inBuf, (uint32_t)currentChunkSize, d_codes, d_blockSizes);

        // 7. GPU-only Prefix Scan
        CUDA_CHECK(cudaEventRecord(scanStart, computeStream));
        gpuScanBlockSizesKernel<<<1, 1, 0, computeStream>>>(d_blockSizes, d_blockOffsets, d_totalCompressedSize, (uint32_t)numBlocks);
        CUDA_CHECK(cudaEventRecord(scanEnd, computeStream));

        // 8. GPU Encoding & Bit Packing Kernel
        CUDA_CHECK(cudaEventRecord(encodeStart, computeStream));
        gpuEncodeBlocksKernel<<<numBlocks, 256, 0, computeStream>>>(d_inBuf, (uint32_t)currentChunkSize, d_codes, d_blockOffsets, d_outBuf);
        CUDA_CHECK(cudaEventRecord(encodeEnd, computeStream));

        // 9. Copy total compressed size and block sizes back to CPU
        uint32_t compressedChunkSize = 0;
        CUDA_CHECK(cudaEventRecord(d2hStart, computeStream));
        CUDA_CHECK(cudaMemcpyAsync(&compressedChunkSize, d_totalCompressedSize, sizeof(uint32_t), cudaMemcpyDeviceToHost, computeStream));
        CUDA_CHECK(cudaMemcpyAsync(h_blockSizes[bufIdx], d_blockSizes, numBlocks * sizeof(uint16_t), cudaMemcpyDeviceToHost, computeStream));
        CUDA_CHECK(cudaStreamSynchronize(computeStream));

        // Copy compressed bitstream data back to CPU
        CUDA_CHECK(cudaMemcpyAsync(h_outBuf[bufIdx], d_outBuf, compressedChunkSize, cudaMemcpyDeviceToHost, computeStream));
        CUDA_CHECK(cudaEventRecord(d2hEnd, computeStream));
        CUDA_CHECK(cudaStreamSynchronize(computeStream));

        // Record CUDA Event timings
        float h2dMs, histMs, scanMs, encodeMs, d2hMs;
        CUDA_CHECK(cudaEventElapsedTime(&h2dMs, h2dStart, h2dEnd));
        CUDA_CHECK(cudaEventElapsedTime(&histMs, histStart, histEnd));
        CUDA_CHECK(cudaEventElapsedTime(&scanMs, scanStart, scanEnd));
        CUDA_CHECK(cudaEventElapsedTime(&encodeMs, encodeStart, encodeEnd));
        CUDA_CHECK(cudaEventElapsedTime(&d2hMs, d2hStart, d2hEnd));

        timings.h2dTime += h2dMs;
        timings.histTime += histMs;
        timings.scanTime += scanMs;
        timings.encodeTime += encodeMs;
        timings.d2hTime += d2hMs;
        timings.totalGpuTime += h2dMs + histMs + scanMs + encodeMs + d2hMs;

        // --- Write compressed output to Disk ---
        ioStart = std::chrono::high_resolution_clock::now();
        outFile.write(reinterpret_cast<const char*>(&compressedChunkSize), sizeof(compressedChunkSize));
        uint32_t uncompressedChunkSizeVal = (uint32_t)currentChunkSize;
        outFile.write(reinterpret_cast<const char*>(&uncompressedChunkSizeVal), sizeof(uncompressedChunkSizeVal));

        for (int i = 0; i < ALPHABET_SIZE; ++i) {
            uint8_t lenByte = (uint8_t)lengths[i];
            outFile.write(reinterpret_cast<const char*>(&lenByte), 1);
        }

        outFile.write(reinterpret_cast<const char*>(h_blockSizes[bufIdx]), numBlocks * sizeof(uint16_t));

        if (compressedChunkSize > 0) {
            outFile.write(reinterpret_cast<const char*>(h_outBuf[bufIdx]), compressedChunkSize);
        }
        ioEnd = std::chrono::high_resolution_clock::now();
        timings.diskWriteTime += std::chrono::duration<double, std::milli>(ioEnd - ioStart).count();

        bytesProcessed += currentChunkSize;
        currentChunkSize = nextChunkSize;
        bufIdx = nextBufIdx;
    }

    // Clean up
    CUDA_CHECK(cudaStreamDestroy(computeStream));
    CUDA_CHECK(cudaEventDestroy(h2dStart));
    CUDA_CHECK(cudaEventDestroy(h2dEnd));
    CUDA_CHECK(cudaEventDestroy(histStart));
    CUDA_CHECK(cudaEventDestroy(histEnd));
    CUDA_CHECK(cudaEventDestroy(scanStart));
    CUDA_CHECK(cudaEventDestroy(scanEnd));
    CUDA_CHECK(cudaEventDestroy(encodeStart));
    CUDA_CHECK(cudaEventDestroy(encodeEnd));
    CUDA_CHECK(cudaEventDestroy(d2hStart));
    CUDA_CHECK(cudaEventDestroy(d2hEnd));

    CUDA_CHECK(cudaFreeHost(h_inBuf[0]));
    CUDA_CHECK(cudaFreeHost(h_inBuf[1]));
    CUDA_CHECK(cudaFreeHost(h_outBuf[0]));
    CUDA_CHECK(cudaFreeHost(h_outBuf[1]));
    CUDA_CHECK(cudaFreeHost(h_blockSizes[0]));
    CUDA_CHECK(cudaFreeHost(h_blockSizes[1]));

    CUDA_CHECK(cudaFree(d_inBuf));
    CUDA_CHECK(cudaFree(d_outBuf));
    CUDA_CHECK(cudaFree(d_freq));
    CUDA_CHECK(cudaFree(d_codes));
    CUDA_CHECK(cudaFree(d_blockSizes));
    CUDA_CHECK(cudaFree(d_blockOffsets));
    CUDA_CHECK(cudaFree(d_totalCompressedSize));
}

// GPU Decompression Streaming Pipeline
void gpuDecompress(const std::string& inputFilePath, const std::string& outputFilePath, GpuTimings& timings) {
    std::ifstream inFile(inputFilePath, std::ios::binary);
    std::ofstream outFile(outputFilePath, std::ios::binary);

    if (!inFile || !outFile) {
        std::cerr << "[GPU] Error opening files for decompression.\n";
        return;
    }

    char magic[4];
    inFile.read(magic, 4);
    if (std::memcmp(magic, "HUFF", 4) != 0) {
        std::cerr << "[GPU] Invalid file format (magic bytes mismatch).\n";
        return;
    }

    uint64_t originalFileSize;
    inFile.read(reinterpret_cast<char*>(&originalFileSize), sizeof(originalFileSize));

    const size_t CHUNK_SIZE = 32 * 1024 * 1024; // 32 MB chunk

    uint8_t* h_compressedBuf[2];
    uint8_t* h_decompressedBuf[2];
    uint16_t* h_blockSizes[2];
    uint32_t* h_blockOffsets[2];

    CUDA_CHECK(cudaMallocHost(&h_compressedBuf[0], CHUNK_SIZE));
    CUDA_CHECK(cudaMallocHost(&h_compressedBuf[1], CHUNK_SIZE));
    CUDA_CHECK(cudaMallocHost(&h_decompressedBuf[0], CHUNK_SIZE));
    CUDA_CHECK(cudaMallocHost(&h_decompressedBuf[1], CHUNK_SIZE));

    size_t maxBlocks = (CHUNK_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CUDA_CHECK(cudaMallocHost(&h_blockSizes[0], maxBlocks * sizeof(uint16_t)));
    CUDA_CHECK(cudaMallocHost(&h_blockSizes[1], maxBlocks * sizeof(uint16_t)));
    CUDA_CHECK(cudaMallocHost(&h_blockOffsets[0], maxBlocks * sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocHost(&h_blockOffsets[1], maxBlocks * sizeof(uint32_t)));

    uint8_t* d_compressed;
    uint8_t* d_decompressed;
    uint16_t* d_blockSizes;
    uint32_t* d_blockOffsets;

    DecEntry* d_lut;
    int16_t* d_left;
    int16_t* d_right;
    uint8_t* d_symbol;

    CUDA_CHECK(cudaMalloc(&d_compressed, CHUNK_SIZE));
    CUDA_CHECK(cudaMalloc(&d_decompressed, CHUNK_SIZE));
    CUDA_CHECK(cudaMalloc(&d_blockSizes, maxBlocks * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_blockOffsets, maxBlocks * sizeof(uint32_t)));

    CUDA_CHECK(cudaMalloc(&d_lut, 256 * sizeof(DecEntry)));
    CUDA_CHECK(cudaMalloc(&d_left, 512 * sizeof(int16_t)));
    CUDA_CHECK(cudaMalloc(&d_right, 512 * sizeof(int16_t)));
    CUDA_CHECK(cudaMalloc(&d_symbol, 512 * sizeof(uint8_t)));

    cudaStream_t decodeStream;
    CUDA_CHECK(cudaStreamCreate(&decodeStream));

    cudaEvent_t h2dStart, h2dEnd;
    cudaEvent_t decodeStart, decodeEnd;
    cudaEvent_t d2hStart, d2hEnd;

    CUDA_CHECK(cudaEventCreate(&h2dStart));
    CUDA_CHECK(cudaEventCreate(&h2dEnd));
    CUDA_CHECK(cudaEventCreate(&decodeStart));
    CUDA_CHECK(cudaEventCreate(&decodeEnd));
    CUDA_CHECK(cudaEventCreate(&d2hStart));
    CUDA_CHECK(cudaEventCreate(&d2hEnd));

    uint64_t bytesDecompressed = 0;
    int bufIdx = 0;

    auto ioStart = std::chrono::high_resolution_clock::now();
    uint32_t currentCompressedSize = 0;
    uint32_t currentUncompressedSize = 0;
    uint32_t lengths[2][ALPHABET_SIZE] = { 0 };

    bool hasChunk = false;
    if (inFile.read(reinterpret_cast<char*>(&currentCompressedSize), sizeof(currentCompressedSize))) {
        inFile.read(reinterpret_cast<char*>(&currentUncompressedSize), sizeof(currentUncompressedSize));

        for (int i = 0; i < ALPHABET_SIZE; ++i) {
            uint8_t lenByte;
            inFile.read(reinterpret_cast<char*>(&lenByte), 1);
            lengths[bufIdx][i] = lenByte;
        }

        size_t numBlocks = (currentUncompressedSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        inFile.read(reinterpret_cast<char*>(h_blockSizes[bufIdx]), numBlocks * sizeof(uint16_t));

        uint32_t offset = 0;
        for (size_t i = 0; i < numBlocks; ++i) {
            h_blockOffsets[bufIdx][i] = offset;
            offset += h_blockSizes[bufIdx][i];
        }

        inFile.read(reinterpret_cast<char*>(h_compressedBuf[bufIdx]), currentCompressedSize);
        hasChunk = true;
    }
    auto ioEnd = std::chrono::high_resolution_clock::now();
    timings.diskReadTime += std::chrono::duration<double, std::milli>(ioEnd - ioStart).count();

    while (hasChunk && bytesDecompressed < originalFileSize) {
        uint32_t nextCompressedSize = 0;
        uint32_t nextUncompressedSize = 0;
        int nextBufIdx = 1 - bufIdx;
        bool nextHasChunk = false;

        ioStart = std::chrono::high_resolution_clock::now();
        if (inFile.read(reinterpret_cast<char*>(&nextCompressedSize), sizeof(nextCompressedSize))) {
            inFile.read(reinterpret_cast<char*>(&nextUncompressedSize), sizeof(nextUncompressedSize));

            for (int i = 0; i < ALPHABET_SIZE; ++i) {
                uint8_t lenByte;
                inFile.read(reinterpret_cast<char*>(&lenByte), 1);
                lengths[nextBufIdx][i] = lenByte;
            }

            size_t numBlocks = (nextUncompressedSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
            inFile.read(reinterpret_cast<char*>(h_blockSizes[nextBufIdx]), numBlocks * sizeof(uint16_t));

            uint32_t offset = 0;
            for (size_t i = 0; i < numBlocks; ++i) {
                h_blockOffsets[nextBufIdx][i] = offset;
                offset += h_blockSizes[nextBufIdx][i];
            }

            inFile.read(reinterpret_cast<char*>(h_compressedBuf[nextBufIdx]), nextCompressedSize);
            nextHasChunk = true;
        }
        ioEnd = std::chrono::high_resolution_clock::now();
        timings.diskReadTime += std::chrono::duration<double, std::milli>(ioEnd - ioStart).count();

        auto cpuTreeStart = std::chrono::high_resolution_clock::now();
        DecEntry h_lut[256];
        std::vector<int16_t> h_left;
        std::vector<int16_t> h_right;
        std::vector<uint8_t> h_symbol;
        buildDecoderTables(lengths[bufIdx], h_lut, h_left, h_right, h_symbol);
        auto cpuTreeEnd = std::chrono::high_resolution_clock::now();
        timings.treeTime += std::chrono::duration<double, std::milli>(cpuTreeEnd - cpuTreeStart).count();

        size_t numBlocks = (currentUncompressedSize + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // 1. Host-to-Device Copy
        CUDA_CHECK(cudaEventRecord(h2dStart, decodeStream));
        CUDA_CHECK(cudaMemcpyAsync(d_compressed, h_compressedBuf[bufIdx], currentCompressedSize, cudaMemcpyHostToDevice, decodeStream));
        CUDA_CHECK(cudaMemcpyAsync(d_blockSizes, h_blockSizes[bufIdx], numBlocks * sizeof(uint16_t), cudaMemcpyHostToDevice, decodeStream));
        CUDA_CHECK(cudaMemcpyAsync(d_blockOffsets, h_blockOffsets[bufIdx], numBlocks * sizeof(uint32_t), cudaMemcpyHostToDevice, decodeStream));

        CUDA_CHECK(cudaMemcpyAsync(d_lut, h_lut, 256 * sizeof(DecEntry), cudaMemcpyHostToDevice, decodeStream));
        CUDA_CHECK(cudaMemcpyAsync(d_left, h_left.data(), h_left.size() * sizeof(int16_t), cudaMemcpyHostToDevice, decodeStream));
        CUDA_CHECK(cudaMemcpyAsync(d_right, h_right.data(), h_right.size() * sizeof(int16_t), cudaMemcpyHostToDevice, decodeStream));
        CUDA_CHECK(cudaMemcpyAsync(d_symbol, h_symbol.data(), h_symbol.size() * sizeof(uint8_t), cudaMemcpyHostToDevice, decodeStream));
        CUDA_CHECK(cudaEventRecord(h2dEnd, decodeStream));

        // 2. Parallel Decoding
        CUDA_CHECK(cudaEventRecord(decodeStart, decodeStream));
        int decodeBlockSize = 256;
        int decodeGridSize = (numBlocks + decodeBlockSize - 1) / decodeBlockSize;
        gpuDecodeBlocksKernel<<<decodeGridSize, decodeBlockSize, 0, decodeStream>>>(
            d_compressed, d_blockOffsets, d_blockSizes,
            d_decompressed, currentUncompressedSize,
            d_lut, d_left, d_right, d_symbol,
            (uint32_t)numBlocks
        );
        CUDA_CHECK(cudaEventRecord(decodeEnd, decodeStream));

        // 3. Device-to-Host Copy
        CUDA_CHECK(cudaEventRecord(d2hStart, decodeStream));
        CUDA_CHECK(cudaMemcpyAsync(h_decompressedBuf[bufIdx], d_decompressed, currentUncompressedSize, cudaMemcpyDeviceToHost, decodeStream));
        CUDA_CHECK(cudaEventRecord(d2hEnd, decodeStream));

        CUDA_CHECK(cudaStreamSynchronize(decodeStream));

        float h2dMs, decodeMs, d2hMs;
        CUDA_CHECK(cudaEventElapsedTime(&h2dMs, h2dStart, h2dEnd));
        CUDA_CHECK(cudaEventElapsedTime(&decodeMs, decodeStart, decodeEnd));
        CUDA_CHECK(cudaEventElapsedTime(&d2hMs, d2hStart, d2hEnd));

        timings.h2dTime += h2dMs;
        timings.encodeTime += decodeMs; // Re-use encodeTime for decompression kernel time
        timings.d2hTime += d2hMs;
        timings.totalGpuTime += h2dMs + decodeMs + d2hMs;

        // --- Write to Disk ---
        ioStart = std::chrono::high_resolution_clock::now();
        outFile.write(reinterpret_cast<const char*>(h_decompressedBuf[bufIdx]), currentUncompressedSize);
        ioEnd = std::chrono::high_resolution_clock::now();
        timings.diskWriteTime += std::chrono::duration<double, std::milli>(ioEnd - ioStart).count();

        bytesDecompressed += currentUncompressedSize;
        hasChunk = nextHasChunk;
        currentCompressedSize = nextCompressedSize;
        currentUncompressedSize = nextUncompressedSize;
        bufIdx = nextBufIdx;
    }

    // Clean up
    CUDA_CHECK(cudaStreamDestroy(decodeStream));
    CUDA_CHECK(cudaEventDestroy(h2dStart));
    CUDA_CHECK(cudaEventDestroy(h2dEnd));
    CUDA_CHECK(cudaEventDestroy(decodeStart));
    CUDA_CHECK(cudaEventDestroy(decodeEnd));
    CUDA_CHECK(cudaEventDestroy(d2hStart));
    CUDA_CHECK(cudaEventDestroy(d2hEnd));

    CUDA_CHECK(cudaFreeHost(h_compressedBuf[0]));
    CUDA_CHECK(cudaFreeHost(h_compressedBuf[1]));
    CUDA_CHECK(cudaFreeHost(h_decompressedBuf[0]));
    CUDA_CHECK(cudaFreeHost(h_decompressedBuf[1]));
    CUDA_CHECK(cudaFreeHost(h_blockSizes[0]));
    CUDA_CHECK(cudaFreeHost(h_blockSizes[1]));
    CUDA_CHECK(cudaFreeHost(h_blockOffsets[0]));
    CUDA_CHECK(cudaFreeHost(h_blockOffsets[1]));

    CUDA_CHECK(cudaFree(d_compressed));
    CUDA_CHECK(cudaFree(d_decompressed));
    CUDA_CHECK(cudaFree(d_blockSizes));
    CUDA_CHECK(cudaFree(d_blockOffsets));
    CUDA_CHECK(cudaFree(d_lut));
    CUDA_CHECK(cudaFree(d_left));
    CUDA_CHECK(cudaFree(d_right));
    CUDA_CHECK(cudaFree(d_symbol));
}
