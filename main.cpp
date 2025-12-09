#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

#define WIDTH 1024
#define HEIGHT 1024
#define STEPS 100

// CUDA kernel to update the grid
__global__ void gameOfLifeKernel(int* current, int* next, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int count = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            int nx = (x + dx + width) % width;   // wrap around
            int ny = (y + dy + height) % height; // wrap around
            count += current[ny * width + nx];
        }
    }

    int idx = y * width + x;
    if (current[idx] == 1 && (count == 2 || count == 3)) {
        next[idx] = 1;
    } else if (current[idx] == 0 && count == 3) {
        next[idx] = 1;
    } else {
        next[idx] = 0;
    }
}

int main() {
    std::vector<int> h_grid(WIDTH * HEIGHT);
    srand(time(0));

    // Initialize grid randomly
    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
        h_grid[i] = rand() % 2;
    }

    int *d_current, *d_next;
    cudaMalloc(&d_current, WIDTH * HEIGHT * sizeof(int));
    cudaMalloc(&d_next, WIDTH * HEIGHT * sizeof(int));

    cudaMemcpy(d_current, h_grid.data(), WIDTH * HEIGHT * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

    for (int step = 0; step < STEPS; ++step) {
        gameOfLifeKernel<<<numBlocks, threadsPerBlock>>>(d_current, d_next, WIDTH, HEIGHT);
        cudaDeviceSynchronize();

        std::swap(d_current, d_next);
    }

    cudaMemcpy(h_grid.data(), d_current, WIDTH * HEIGHT * sizeof(int), cudaMemcpyDeviceToHost);

    // Print small part of the grid
    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 10; ++x) {
            std::cout << h_grid[y * WIDTH + x] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_current);
    cudaFree(d_next);

    return 0;
}
