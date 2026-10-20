#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cstdint>
#include <string>

// Maximum alphabet size (256 for standard byte-level data)
#define ALPHABET_SIZE 256

// Block size for parallel processing (16 KB)
#define BLOCK_SIZE (16 * 1024)

// Canonical Huffman code representation
struct HuffmanCode {
    uint64_t code;   // Bit pattern
    uint32_t length; // Bit length of the code
};

// Decoder Lookup Table Entry (8-bit prefix)
struct DecEntry {
    uint8_t symbol;
    uint8_t len;     // Code length (0 if length > 8, fallback to tree)
};

// Struct to hold detailed GPU benchmark timings
struct GpuTimings {
    double h2dTime = 0.0;          // Host-to-Device transfer time (ms)
    double histTime = 0.0;         // Histogram kernel time (ms)
    double scanTime = 0.0;         // Prefix scan kernel time (ms)
    double encodeTime = 0.0;       // Encoding kernel time (ms)
    double d2hTime = 0.0;          // Device-to-Host transfer time (ms)
    double totalGpuTime = 0.0;     // Total time on GPU (ms)
    double treeTime = 0.0;         // Tree building and canonical code gen time (ms)
    double diskReadTime = 0.0;     // Disk read I/O time (ms)
    double diskWriteTime = 0.0;    // Disk write I/O time (ms)
};

// CPU-only Huffman Compression & Decompression
void cpuCompress(const std::string& inputFilePath, const std::string& outputFilePath);
void cpuDecompress(const std::string& inputFilePath, const std::string& outputFilePath);

// GPU-only Huffman Compression & Decompression
void gpuCompress(const std::string& inputFilePath, const std::string& outputFilePath, GpuTimings& timings);
void gpuDecompress(const std::string& inputFilePath, const std::string& outputFilePath, GpuTimings& timings);

#endif // KERNELS_CUH
