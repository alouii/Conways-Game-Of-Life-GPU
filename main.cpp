#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cstdint>

#define WIDTH 1024       // Must be multiple of 32 for simplicity
#define HEIGHT 1024
#define STEPS 100

using Word = uint32_t;
constexpr int WORD_SIZE = 32;

// Helper to get bit
__device__ inline int getBit(Word w, int pos) {
    return (w >> pos) & 1;
}

// Helper to set bit
__device__ inline void setBit(Word &w, int pos, int val) {
    w &= ~(1u << pos);
    w |= (val << pos);
}

// CUDA kernel for bit-packed Game of Life
__global__ void gameOfLifeBitPacked(const Word* current, Word* next, int widthWords, int height) {
    int xWord = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (xWord >= widthWords || y >= height) return;

    Word newWord = 0;

    for (int bit = 0; bit < WORD_SIZE; ++bit) {
        int x = xWord * WORD_SIZE + bit;

        int count = 0;
        // iterate over neighbors
        for (int dy = -1; dy <= 1; ++dy) {
            int ny = (y + dy + height) % height;
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                int nx = (x + dx + WIDTH) % WIDTH;
                int nWord = nx / WORD_SIZE;
                int nBit = nx % WORD_SIZE;
                count += getBit(current[ny * widthWords + nWord], nBit);
            }
        }

        int idx = y * widthWords + xWord;
        int currentCell = getBit(current[idx], bit);
        int newCell = (currentCell && (count == 2 || count == 3)) || (!currentCell && count == 3);
        setBit(newWord, bit, newCell);
    }

    next[y * widthWords + xWord] = newWord;
}

int main() {
    int widthWords = WIDTH / WORD_SIZE;
    std::vector<Word> h_grid(WIDTH / WORD_SIZE * HEIGHT);
    srand(time(0));

    // Random initialization
    for (auto &w : h_grid) w = rand();

    Word *d_current, *d_next;
    cudaMalloc(&d_current, h_grid.size() * sizeof(Word));
    cudaMalloc(&d_next, h_grid.size() * sizeof(Word));
    cudaMemcpy(d_current, h_grid.data(), h_grid.size() * sizeof(Word), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((widthWords + 15) / 16, (HEIGHT + 15) / 16);

    for (int step = 0; step < STEPS; ++step) {
        gameOfLifeBitPacked<<<numBlocks, threadsPerBlock>>>(d_current, d_next, widthWords, HEIGHT);
        cudaDeviceSynchronize();
        std::swap(d_current, d_next);
    }

    cudaMemcpy(h_grid.data(), d_current, h_grid.size() * sizeof(Word), cudaMemcpyDeviceToHost);

    // Print first 64 cells
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 64; ++x) {
            int w = x / WORD_SIZE;
            int b = x % WORD_SIZE;
            std::cout << ((h_grid[y * widthWords + w] >> b) & 1);
        }
        std::cout << std::endl;
    }

    cudaFree(d_current);
    cudaFree(d_next);
    return 0;
}
