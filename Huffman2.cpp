#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <windows.h>
#include <commdlg.h>
#include "kernels.cuh"

// Helper to open standard Windows file dialog
std::string openFileDialog() {
    OPENFILENAME ofn;
    CHAR szFile[260] = { 0 };

    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFile = szFile;
    ofn.lpstrFile[0] = '\0';
    ofn.nMaxFile = sizeof(szFile);
    ofn.lpstrFilter = "All Files\0*.*\0Text Files\0*.TXT\0";
    ofn.nFilterIndex = 1;
    ofn.lpstrFileTitle = NULL;
    ofn.nMaxFileTitle = 0;
    ofn.lpstrInitialDir = NULL;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

    if (GetOpenFileName(&ofn)) {
        return std::string(ofn.lpstrFile);
    }
    return "";
}

// Generate structured, compressible data for testing Huffman coding
void generateDummyFile(const std::string& path, uint64_t size) {
    std::cout << "Generating " << size / (1024 * 1024) << " MB test file at: " << path << "...\n";
    std::ofstream outFile(path, std::ios::binary);
    if (!outFile) {
        std::cerr << "Failed to create test file.\n";
        return;
    }

    // A frequency distribution of characters resembling English text + spaces
    std::vector<char> pattern = {
        'e', 'e', 'e', 'e', 'e', 't', 't', 't', 't', 'a', 'a', 'a', 'o', 'o', 'o',
        'i', 'i', 'n', 'n', 's', 's', 'h', 'r', 'd', 'l', 'c', 'u', 'm', 'w', 'f',
        'g', 'y', 'p', 'b', 'v', 'k', 'j', 'x', 'q', 'z', '\n', ' ', ' ', ' ', ' '
    };

    size_t patternSize = pattern.size();
    std::vector<char> buffer(1024 * 1024); // 1 MB buffer
    uint32_t seed = 42;
    for (size_t i = 0; i < buffer.size(); ++i) {
        seed = seed * 1664525 + 1013904223;
        buffer[i] = pattern[seed % patternSize];
    }

    uint64_t bytesWritten = 0;
    while (bytesWritten < size) {
        uint64_t toWrite = std::min(size - bytesWritten, (uint64_t)buffer.size());
        outFile.write(buffer.data(), toWrite);
        bytesWritten += toWrite;
    }
    std::cout << "Test file generated successfully.\n";
}

// Compare two files byte-by-byte
bool verifyCorrectness(const std::string& originalPath, const std::string& decompressedPath, uint64_t& mismatchOffset) {
    std::ifstream origFile(originalPath, std::ios::binary);
    std::ifstream decFile(decompressedPath, std::ios::binary);

    if (!origFile || !decFile) {
        std::cerr << "Error opening files for verification.\n";
        return false;
    }

    origFile.seekg(0, std::ios::end);
    uint64_t origSize = origFile.tellg();
    origFile.seekg(0, std::ios::beg);

    decFile.seekg(0, std::ios::end);
    uint64_t decSize = decFile.tellg();
    decFile.seekg(0, std::ios::beg);

    if (origSize != decSize) {
        std::cerr << "Size mismatch: Original = " << origSize << " bytes, Decompressed = " << decSize << " bytes.\n";
        return false;
    }

    const size_t BUF_SIZE = 64 * 1024;
    std::vector<char> origBuf(BUF_SIZE);
    std::vector<char> decBuf(BUF_SIZE);

    uint64_t offset = 0;
    while (offset < origSize) {
        size_t currentChunk = std::min((uint64_t)BUF_SIZE, origSize - offset);
        origFile.read(origBuf.data(), currentChunk);
        decFile.read(decBuf.data(), currentChunk);

        if (std::memcmp(origBuf.data(), decBuf.data(), currentChunk) != 0) {
            for (size_t i = 0; i < currentChunk; ++i) {
                if (origBuf[i] != decBuf[i]) {
                    mismatchOffset = offset + i;
                    return false;
                }
            }
        }
        offset += currentChunk;
    }

    return true;
}

int main() {
    std::cout << "=========================================================\n";
    std::cout << "   CPU vs GPU Canonical Huffman Compression Benchmark    \n";
    std::cout << "=========================================================\n\n";

    std::string inputFilePath = "";

    std::cout << "Choose input file source:\n";
    std::cout << "1. Select an existing file from disk\n";
    std::cout << "2. Generate a new compressible test file (10 MB)\n";
    std::cout << "3. Generate a new compressible test file (100 MB)\n";
    std::cout << "4. Generate a new compressible test file (500 MB)\n";
    std::cout << "5. Generate a new compressible test file (1 GB)\n";
    std::cout << "Select option (1-5): ";
    
    int option;
    if (!(std::cin >> option)) {
        std::cerr << "Invalid option selected.\n";
        return 1;
    }

    if (option == 1) {
        std::cout << "Please select the file via dialog...\n";
        inputFilePath = openFileDialog();
        if (inputFilePath.empty()) {
            std::cerr << "No file selected. Exiting.\n";
            return 1;
        }
    } else {
        uint64_t size = 10ULL * 1024 * 1024; // Default 10 MB
        if (option == 3) size = 100ULL * 1024 * 1024;
        else if (option == 4) size = 500ULL * 1024 * 1024;
        else if (option == 5) size = 1024ULL * 1024 * 1024;

        inputFilePath = "benchmark_test.bin";
        generateDummyFile(inputFilePath, size);
    }

    std::ifstream testFile(inputFilePath, std::ios::binary);
    if (!testFile) {
        std::cerr << "Error opening file: " << inputFilePath << "\n";
        return 1;
    }
    testFile.seekg(0, std::ios::end);
    uint64_t fileSize = testFile.tellg();
    testFile.close();

    double fileSizeMB = (double)fileSize / (1024.0 * 1024.0);
    std::cout << "File selected: " << inputFilePath << " (" << std::fixed << std::setprecision(2) << fileSizeMB << " MB)\n\n";

    // Auto-generate output paths
    std::string cpuCompressedPath = inputFilePath + ".cpu.huff";
    std::string gpuCompressedPath = inputFilePath + ".gpu.huff";
    std::string cpuDecompressedPath = inputFilePath + ".cpu.dec";
    std::string gpuDecompressedPath = inputFilePath + ".gpu.dec";

    // ---------------------------------------------------------
    // CPU PIPELINE
    // ---------------------------------------------------------
    std::cout << "[CPU Pipeline] Starting sequential compression...\n";
    auto cpuCompStart = std::chrono::high_resolution_clock::now();
    cpuCompress(inputFilePath, cpuCompressedPath);
    auto cpuCompEnd = std::chrono::high_resolution_clock::now();
    double cpuCompTimeMs = std::chrono::duration<double, std::milli>(cpuCompEnd - cpuCompStart).count();
    std::cout << "[CPU Pipeline] Compression completed in " << cpuCompTimeMs << " ms.\n";

    std::cout << "[CPU Pipeline] Starting sequential decompression...\n";
    auto cpuDecStart = std::chrono::high_resolution_clock::now();
    cpuDecompress(cpuCompressedPath, cpuDecompressedPath);
    auto cpuDecEnd = std::chrono::high_resolution_clock::now();
    double cpuDecTimeMs = std::chrono::duration<double, std::milli>(cpuDecEnd - cpuDecStart).count();
    std::cout << "[CPU Pipeline] Decompression completed in " << cpuDecTimeMs << " ms.\n\n";

    // ---------------------------------------------------------
    // GPU PIPELINE
    // ---------------------------------------------------------
    std::cout << "[GPU Pipeline] Starting parallel compression...\n";
    GpuTimings gpuCompTimings;
    auto gpuCompStart = std::chrono::high_resolution_clock::now();
    gpuCompress(inputFilePath, gpuCompressedPath, gpuCompTimings);
    auto gpuCompEnd = std::chrono::high_resolution_clock::now();
    double totalGpuCompTimeMs = std::chrono::duration<double, std::milli>(gpuCompEnd - gpuCompStart).count();
    std::cout << "[GPU Pipeline] Compression completed in " << totalGpuCompTimeMs << " ms.\n";

    std::cout << "[GPU Pipeline] Starting parallel decompression...\n";
    GpuTimings gpuDecTimings;
    auto gpuDecStart = std::chrono::high_resolution_clock::now();
    gpuDecompress(gpuCompressedPath, gpuDecompressedPath, gpuDecTimings);
    auto gpuDecEnd = std::chrono::high_resolution_clock::now();
    double totalGpuDecTimeMs = std::chrono::duration<double, std::milli>(gpuDecEnd - gpuDecStart).count();
    std::cout << "[GPU Pipeline] Decompression completed in " << totalGpuDecTimeMs << " ms.\n\n";

    // ---------------------------------------------------------
    // CORRECTNESS VERIFICATION
    // ---------------------------------------------------------
    std::cout << "---------------------------------------------------------\n";
    std::cout << "                  Correctness Validation                 \n";
    std::cout << "---------------------------------------------------------\n";
    
    uint64_t cpuMismatchOffset = 0;
    bool cpuCorrect = verifyCorrectness(inputFilePath, cpuDecompressedPath, cpuMismatchOffset);
    if (cpuCorrect) {
        std::cout << "[ SUCCESS ] CPU decompressed output matches original byte-for-byte!\n";
    } else {
        std::cout << "[ FAILURE ] CPU decompression mismatch at byte offset: " << cpuMismatchOffset << "\n";
    }

    uint64_t gpuMismatchOffset = 0;
    bool gpuCorrect = verifyCorrectness(inputFilePath, gpuDecompressedPath, gpuMismatchOffset);
    if (gpuCorrect) {
        std::cout << "[ SUCCESS ] GPU decompressed output matches original byte-for-byte!\n";
    } else {
        std::cout << "[ FAILURE ] GPU decompression mismatch at byte offset: " << gpuMismatchOffset << "\n";
    }

    // Check sizes
    std::ifstream cpuHuffFile(cpuCompressedPath, std::ios::binary | std::ios::ate);
    uint64_t cpuHuffSize = cpuHuffFile.tellg();
    cpuHuffFile.close();

    std::ifstream gpuHuffFile(gpuCompressedPath, std::ios::binary | std::ios::ate);
    uint64_t gpuHuffSize = gpuHuffFile.tellg();
    gpuHuffFile.close();

    double cpuRatio = (double)fileSize / cpuHuffSize;
    double gpuRatio = (double)fileSize / gpuHuffSize;

    // ---------------------------------------------------------
    // PERFORMANCE REPORT
    // ---------------------------------------------------------
    std::cout << "\n---------------------------------------------------------\n";
    std::cout << "                Performance Comparison Summary           \n";
    std::cout << "---------------------------------------------------------\n";
    std::cout << std::left << std::setw(28) << "Metric" 
              << std::setw(15) << "CPU Only" 
              << std::setw(15) << "GPU Pipeline" << "\n";
    std::cout << "---------------------------------------------------------\n";
    std::cout << std::left << std::setw(28) << "File Size (MB)" 
              << std::setw(15) << std::fixed << std::setprecision(2) << fileSizeMB 
              << std::setw(15) << fileSizeMB << "\n";
    std::cout << std::left << std::setw(28) << "Compressed Size (MB)" 
              << std::setw(15) << std::fixed << std::setprecision(2) << ((double)cpuHuffSize / (1024.0 * 1024.0))
              << std::setw(15) << ((double)gpuHuffSize / (1024.0 * 1024.0)) << "\n";
    std::cout << std::left << std::setw(28) << "Compression Ratio" 
              << std::setw(15) << std::fixed << std::setprecision(3) << cpuRatio 
              << std::setw(15) << gpuRatio << "\n";
    std::cout << std::left << std::setw(28) << "Compression Time (ms)" 
              << std::setw(15) << std::fixed << std::setprecision(2) << cpuCompTimeMs 
              << std::setw(15) << totalGpuCompTimeMs << "\n";
    std::cout << std::left << std::setw(28) << "Compression Speed (MB/s)" 
              << std::setw(15) << std::fixed << std::setprecision(2) << (fileSizeMB / (cpuCompTimeMs / 1000.0))
              << std::setw(15) << (fileSizeMB / (totalGpuCompTimeMs / 1000.0)) << "\n";
    std::cout << std::left << std::setw(28) << "Decompression Time (ms)" 
              << std::setw(15) << std::fixed << std::setprecision(2) << cpuDecTimeMs 
              << std::setw(15) << totalGpuDecTimeMs << "\n";
    std::cout << std::left << std::setw(28) << "Decompression Speed (MB/s)" 
              << std::setw(15) << std::fixed << std::setprecision(2) << (fileSizeMB / (cpuDecTimeMs / 1000.0))
              << std::setw(15) << (fileSizeMB / (totalGpuDecTimeMs / 1000.0)) << "\n";
    
    std::cout << "---------------------------------------------------------\n";
    std::cout << std::left << std::setw(28) << "COMPRESSION SPEEDUP" 
              << std::fixed << std::setprecision(2) << (cpuCompTimeMs / totalGpuCompTimeMs) << "x\n";
    std::cout << std::left << std::setw(28) << "DECOMPRESSION SPEEDUP" 
              << std::fixed << std::setprecision(2) << (cpuDecTimeMs / totalGpuDecTimeMs) << "x\n";

    // Detailed GPU Profiling Breakdown
    std::cout << "\n---------------------------------------------------------\n";
    std::cout << "               Detailed GPU Timing Breakdown             \n";
    std::cout << "---------------------------------------------------------\n";
    std::cout << std::left << std::setw(35) << "Stage" 
              << std::setw(20) << "Compression (ms)" 
              << std::setw(20) << "Decompression (ms)" << "\n";
    std::cout << "---------------------------------------------------------\n";
    std::cout << std::left << std::setw(35) << "Disk Read I/O" 
              << std::setw(20) << std::fixed << std::setprecision(2) << gpuCompTimings.diskReadTime 
              << std::setw(20) << gpuDecTimings.diskReadTime << "\n";
    std::cout << std::left << std::setw(35) << "PCIe Host-to-Device Copy (H2D)" 
              << std::setw(20) << std::fixed << std::setprecision(2) << gpuCompTimings.h2dTime 
              << std::setw(20) << gpuDecTimings.h2dTime << "\n";
    std::cout << std::left << std::setw(35) << "GPU Histogram Kernel" 
              << std::setw(20) << std::fixed << std::setprecision(2) << gpuCompTimings.histTime 
              << std::setw(20) << "N/A" << "\n";
    std::cout << std::left << std::setw(35) << "CPU Canonical Book Generation" 
              << std::setw(20) << std::fixed << std::setprecision(2) << gpuCompTimings.treeTime 
              << std::setw(20) << gpuDecTimings.treeTime << "\n";
    std::cout << std::left << std::setw(35) << "GPU Exclusive Prefix Scan" 
              << std::setw(20) << std::fixed << std::setprecision(2) << gpuCompTimings.scanTime 
              << std::setw(20) << "N/A" << "\n";
    std::cout << std::left << std::setw(35) << "GPU Core Kernel (Encode/Decode)" 
              << std::setw(20) << std::fixed << std::setprecision(2) << gpuCompTimings.encodeTime 
              << std::setw(20) << gpuDecTimings.encodeTime << "\n";
    std::cout << std::left << std::setw(35) << "PCIe Device-to-Host Copy (D2H)" 
              << std::setw(20) << std::fixed << std::setprecision(2) << gpuCompTimings.d2hTime 
              << std::setw(20) << gpuDecTimings.d2hTime << "\n";
    std::cout << std::left << std::setw(35) << "Disk Write I/O" 
              << std::setw(20) << std::fixed << std::setprecision(2) << gpuCompTimings.diskWriteTime 
              << std::setw(20) << gpuDecTimings.diskWriteTime << "\n";
    std::cout << "---------------------------------------------------------\n";
    std::cout << std::left << std::setw(35) << "Total Core GPU Time (transfer+kernel)" 
              << std::setw(20) << std::fixed << std::setprecision(2) << gpuCompTimings.totalGpuTime 
              << std::setw(20) << gpuDecTimings.totalGpuTime << "\n";
    std::cout << std::left << std::setw(35) << "Total Pipeline Elapsed Time (ms)" 
              << std::setw(20) << std::fixed << std::setprecision(2) << totalGpuCompTimeMs 
              << std::setw(20) << totalGpuDecTimeMs << "\n";
    std::cout << "=========================================================\n";

    return 0;
}
