#include "kernels.cuh"
#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <algorithm>
#include <cstring>

// Node structure for Huffman tree
struct CpuNode {
    uint8_t symbol;
    uint64_t freq;
    CpuNode* left;
    CpuNode* right;

    CpuNode(uint8_t sym, uint64_t f) : symbol(sym), freq(f), left(nullptr), right(nullptr) {}
    ~CpuNode() {
        delete left;
        delete right;
    }
};

// Comparator for priority queue
struct CompareCpuNode {
    bool operator()(CpuNode* l, CpuNode* r) {
        return l->freq > r->freq;
    }
};

// Recursive helper to extract code lengths
static void getLengths(CpuNode* root, uint32_t depth, uint32_t* lengths) {
    if (!root) return;
    if (!root->left && !root->right) {
        lengths[root->symbol] = (depth == 0) ? 1 : depth; // Handle single symbol case
        return;
    }
    getLengths(root->left, depth + 1, lengths);
    getLengths(root->right, depth + 1, lengths);
}

// Generate Canonical Huffman codes from lengths
static void generateCanonicalCodes(const uint32_t* lengths, HuffmanCode* codes) {
    struct SymbolLength {
        uint8_t symbol;
        uint32_t length;
    };

    std::vector<SymbolLength> activeSymbols;
    for (int i = 0; i < ALPHABET_SIZE; ++i) {
        if (lengths[i] > 0) {
            activeSymbols.push_back({ (uint8_t)i, lengths[i] });
            codes[i] = { 0, 0 }; // Initialize
        } else {
            codes[i] = { 0, 0 };
        }
    }

    // Sort by length first, then by symbol value
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

// Helper structure to write packed bits
struct CpuBitWriter {
    std::vector<uint8_t>& buffer;
    uint64_t bit_buf = 0;
    int bit_count = 0;

    CpuBitWriter(std::vector<uint8_t>& buf) : buffer(buf) {}

    void writeBits(uint64_t code, int length) {
        if (length == 0) return;
        code &= (1ULL << length) - 1; // Mask to avoid garbage

        if (bit_count + length <= 64) {
            bit_buf = (bit_buf << length) | code;
            bit_count += length;
        } else {
            int bits_fit = 64 - bit_count;
            uint64_t first_part = code >> (length - bits_fit);
            bit_buf = (bit_buf << bits_fit) | first_part;

            // Flush 8 bytes (64 bits)
            for (int i = 7; i >= 0; --i) {
                buffer.push_back((uint8_t)(bit_buf >> (i * 8)));
            }

            int remaining_bits = length - bits_fit;
            bit_buf = code & ((1ULL << remaining_bits) - 1);
            bit_count = remaining_bits;
        }
    }

    void flush() {
        if (bit_count > 0) {
            // Byte align
            int align_bits = (8 - (bit_count % 8)) % 8;
            bit_buf <<= align_bits;
            bit_count += align_bits;

            int bytes = bit_count / 8;
            for (int i = bytes - 1; i >= 0; --i) {
                buffer.push_back((uint8_t)(bit_buf >> (i * 8)));
            }
            bit_buf = 0;
            bit_count = 0;
        }
    }
};

// Helper structure to read packed bits
struct CpuBitReader {
    const uint8_t* buffer;
    size_t size;
    size_t byte_idx = 0;
    uint64_t bit_buf = 0;
    int bit_count = 0;

    CpuBitReader(const uint8_t* buf, size_t sz) : buffer(buf), size(sz) {}

    void fill() {
        while (bit_count <= 56 && byte_idx < size) {
            bit_buf = (bit_buf << 8) | buffer[byte_idx++];
            bit_count += 8;
        }
    }

    uint32_t readBit() {
        if (bit_count == 0) {
            fill();
            if (bit_count == 0) return 0;
        }
        uint32_t bit = (bit_buf >> (bit_count - 1)) & 1;
        bit_count--;
        return bit;
    }
};

// Reconstruct Huffman tree from canonical code lengths
static CpuNode* rebuildCanonicalTree(const uint32_t* lengths) {
    HuffmanCode codes[ALPHABET_SIZE];
    generateCanonicalCodes(lengths, codes);

    CpuNode* root = new CpuNode(0, 0);

    for (int i = 0; i < ALPHABET_SIZE; ++i) {
        if (lengths[i] > 0) {
            uint64_t code = codes[i].code;
            uint32_t len = codes[i].length;

            CpuNode* curr = root;
            for (int bitIdx = (int)len - 1; bitIdx >= 0; --bitIdx) {
                uint32_t bit = (code >> bitIdx) & 1;
                if (bit == 0) {
                    if (!curr->left) curr->left = new CpuNode(0, 0);
                    curr = curr->left;
                } else {
                    if (!curr->right) curr->right = new CpuNode(0, 0);
                    curr = curr->right;
                }
            }
            curr->symbol = (uint8_t)i;
        }
    }
    return root;
}

// CPU Compression Routine
void cpuCompress(const std::string& inputFilePath, const std::string& outputFilePath) {
    std::ifstream inFile(inputFilePath, std::ios::binary);
    std::ofstream outFile(outputFilePath, std::ios::binary);

    if (!inFile || !outFile) {
        std::cerr << "[CPU] Error opening files for compression.\n";
        return;
    }

    // Write file header
    const char magic[4] = { 'H', 'U', 'F', 'F' };
    outFile.write(magic, 4);

    inFile.seekg(0, std::ios::end);
    uint64_t originalFileSize = inFile.tellg();
    inFile.seekg(0, std::ios::beg);

    outFile.write(reinterpret_cast<const char*>(&originalFileSize), sizeof(originalFileSize));

    const size_t CHUNK_SIZE = 32 * 1024 * 1024; // 32 MB
    std::vector<uint8_t> inBuf(CHUNK_SIZE);

    uint64_t bytesProcessed = 0;
    while (bytesProcessed < originalFileSize) {
        size_t currentChunkSize = std::min(originalFileSize - bytesProcessed, (uint64_t)CHUNK_SIZE);
        inFile.read(reinterpret_cast<char*>(inBuf.data()), currentChunkSize);

        // 1. Frequency counting
        uint64_t freq[ALPHABET_SIZE] = { 0 };
        for (size_t i = 0; i < currentChunkSize; ++i) {
            freq[inBuf[i]]++;
        }

        // 2. Build Huffman tree on CPU
        std::priority_queue<CpuNode*, std::vector<CpuNode*>, CompareCpuNode> minHeap;
        for (int i = 0; i < ALPHABET_SIZE; ++i) {
            if (freq[i] > 0) {
                minHeap.push(new CpuNode((uint8_t)i, freq[i]));
            }
        }

        uint32_t lengths[ALPHABET_SIZE] = { 0 };
        if (!minHeap.empty()) {
            CpuNode* root = nullptr;
            if (minHeap.size() == 1) {
                CpuNode* single = minHeap.top(); minHeap.pop();
                root = new CpuNode(0, single->freq);
                root->left = single;
            } else {
                while (minHeap.size() > 1) {
                    CpuNode* left = minHeap.top(); minHeap.pop();
                    CpuNode* right = minHeap.top(); minHeap.pop();
                    CpuNode* parent = new CpuNode(0, left->freq + right->freq);
                    parent->left = left;
                    parent->right = right;
                    minHeap.push(parent);
                }
                root = minHeap.top(); minHeap.pop();
            }

            // 3. Generate lengths and codes
            getLengths(root, 0, lengths);
            delete root;
        }

        HuffmanCode codes[ALPHABET_SIZE];
        generateCanonicalCodes(lengths, codes);

        // 4. Compress block-by-block (64 KB)
        size_t numBlocks = (currentChunkSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        std::vector<uint16_t> blockSizes(numBlocks, 0);
        std::vector<std::vector<uint8_t>> blockData(numBlocks);

        for (size_t blockIdx = 0; blockIdx < numBlocks; ++blockIdx) {
            size_t blockOffset = blockIdx * BLOCK_SIZE;
            size_t currentBlockSize = std::min((size_t)BLOCK_SIZE, currentChunkSize - blockOffset);

            CpuBitWriter writer(blockData[blockIdx]);
            for (size_t i = 0; i < currentBlockSize; ++i) {
                uint8_t sym = inBuf[blockOffset + i];
                writer.writeBits(codes[sym].code, codes[sym].length);
            }
            writer.flush();
            blockSizes[blockIdx] = (uint16_t)blockData[blockIdx].size();
        }

        // Calculate total compressed size of this chunk
        uint32_t compressedChunkSize = 0;
        for (const auto& bData : blockData) {
            compressedChunkSize += bData.size();
        }

        // Write Chunk Header
        outFile.write(reinterpret_cast<const char*>(&compressedChunkSize), sizeof(compressedChunkSize));
        uint32_t uncompressedChunkSizeVal = (uint32_t)currentChunkSize;
        outFile.write(reinterpret_cast<const char*>(&uncompressedChunkSizeVal), sizeof(uncompressedChunkSizeVal));

        // Write 256-byte code lengths
        for (int i = 0; i < ALPHABET_SIZE; ++i) {
            uint8_t lenByte = (uint8_t)lengths[i];
            outFile.write(reinterpret_cast<const char*>(&lenByte), 1);
        }

        // Write block sizes array
        outFile.write(reinterpret_cast<const char*>(blockSizes.data()), numBlocks * sizeof(uint16_t));

        // Write contiguous compressed block bitstreams
        for (const auto& bData : blockData) {
            if (!bData.empty()) {
                outFile.write(reinterpret_cast<const char*>(bData.data()), bData.size());
            }
        }

        bytesProcessed += currentChunkSize;
    }
}

// CPU Decompression Routine
void cpuDecompress(const std::string& inputFilePath, const std::string& outputFilePath) {
    std::ifstream inFile(inputFilePath, std::ios::binary);
    std::ofstream outFile(outputFilePath, std::ios::binary);

    if (!inFile || !outFile) {
        std::cerr << "[CPU] Error opening files for decompression.\n";
        return;
    }

    // Verify magic bytes
    char magic[4];
    inFile.read(magic, 4);
    if (std::memcmp(magic, "HUFF", 4) != 0) {
        std::cerr << "[CPU] Invalid file format (magic bytes mismatch).\n";
        return;
    }

    uint64_t originalFileSize;
    inFile.read(reinterpret_cast<char*>(&originalFileSize), sizeof(originalFileSize));

    uint64_t bytesDecompressed = 0;
    while (bytesDecompressed < originalFileSize) {
        uint32_t compressedChunkSize;
        uint32_t uncompressedChunkSize;

        if (!inFile.read(reinterpret_cast<char*>(&compressedChunkSize), sizeof(compressedChunkSize))) {
            break; // EOF
        }
        inFile.read(reinterpret_cast<char*>(&uncompressedChunkSize), sizeof(uncompressedChunkSize));

        // Read code lengths (256 bytes)
        uint32_t lengths[ALPHABET_SIZE] = { 0 };
        for (int i = 0; i < ALPHABET_SIZE; ++i) {
            uint8_t lenByte;
            inFile.read(reinterpret_cast<char*>(&lenByte), 1);
            lengths[i] = lenByte;
        }

        // Rebuild Canonical tree
        CpuNode* root = rebuildCanonicalTree(lengths);

        // Read block sizes array
        size_t numBlocks = (uncompressedChunkSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        std::vector<uint16_t> blockSizes(numBlocks);
        inFile.read(reinterpret_cast<char*>(blockSizes.data()), numBlocks * sizeof(uint16_t));

        // Read entire chunk compressed data
        std::vector<uint8_t> compressedData(compressedChunkSize);
        inFile.read(reinterpret_cast<char*>(compressedData.data()), compressedChunkSize);

        // Decompress block-by-block
        std::vector<uint8_t> decompressedChunk(uncompressedChunkSize);
        size_t compressedDataOffset = 0;

        for (size_t blockIdx = 0; blockIdx < numBlocks; ++blockIdx) {
            size_t blockOutOffset = blockIdx * BLOCK_SIZE;
            size_t currentBlockSize = std::min((size_t)BLOCK_SIZE, (size_t)uncompressedChunkSize - blockOutOffset);
            uint16_t compressedBlockSize = blockSizes[blockIdx];

            CpuBitReader reader(compressedData.data() + compressedDataOffset, compressedBlockSize);
            compressedDataOffset += compressedBlockSize;

            for (size_t i = 0; i < currentBlockSize; ++i) {
                CpuNode* curr = root;
                while (curr->left || curr->right) {
                    uint32_t bit = reader.readBit();
                    if (bit == 0) {
                        curr = curr->left;
                    } else {
                        curr = curr->right;
                    }
                    if (!curr) {
                        std::cerr << "[CPU] Error decoding Huffman stream: invalid tree path.\n";
                        delete root;
                        return;
                    }
                }
                decompressedChunk[blockOutOffset + i] = curr->symbol;
            }
        }

        outFile.write(reinterpret_cast<const char*>(decompressedChunk.data()), uncompressedChunkSize);
        bytesDecompressed += uncompressedChunkSize;
        delete root;
    }
}
