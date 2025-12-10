#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

#define WIDTH 1024
#define HEIGHT 1024
#define STEPS 100
#define TILE_SIZE 16  // Block size

// CUDA kernel using shared memory
__global__ void gameOfLifeShared(int* current, int* next, int width, int height) {
    __shared__ int tile[TILE_SIZE + 2][TILE_SIZE + 2]; // +2 for halo cells

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    // Load cells into shared memory with wrap-around
    int xm1 = (x - 1 + width) % width;
    int xp1 = (x + 1) % width;
    int ym1 = (y - 1 + height) % height;
    int yp1 = (y + 1) % height;

    if (x < width && y < height) {
        tile[ty + 1][tx + 1] = current[y * width + x];

        // Load halo cells
        if (tx == 0) tile[ty + 1][0] = current[y * width + xm1];
        if (tx == TILE_SIZE - 1) tile[ty + 1][TILE_SIZE + 1] = current[y * width + xp1];
        if (ty == 0) tile[0][tx + 1] = current[ym1 * width + x];
        if (ty == TILE_SIZE - 1) tile[TILE_SIZE + 1][tx + 1] = current[yp1 * width + x];

        // Corner halos
        if (tx == 0 && ty == 0) tile[0][0] = current[ym1 * width + xm1];
        if (tx == TILE_SIZE - 1 && ty == 0) tile[0][TILE_SIZE + 1] = current[ym1 * width + xp1];
        if (tx == 0 && ty == TILE_SIZE - 1) tile[TILE_SIZE + 1][0] = current[yp1 * width + xm1];
        if (tx == TILE_SIZE - 1 && ty == TILE_SIZE - 1) tile[TILE_SIZE + 1][TILE_SIZE + 1] = current[yp1 * width + xp1];
    }

    __syncthreads();

    if (x >= width || y >= height) return;

    // Count neighbors
    int count = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            count += tile[ty + 1 + dy][tx + 1 + dx];
        }
    }

    int idx = y * width + x;
    if (tile[ty + 1][tx + 1] == 1 && (count == 2 || count == 3)) next[idx] = 1;
    else if (tile[ty + 1][tx + 1] == 0 && count == 3) next[idx] = 1;
    else next[idx] = 0;
}

int main() {
    std::vector<int> h_grid(WIDTH * HEIGHT);
    srand(time(0));

    // Initialize grid randomly
    for (int i = 0; i < WIDTH * HEIGHT; ++i)
        h_grid[i] = rand() % 2;

    int *d_current, *d_next;
    cudaMalloc(&d_current, WIDTH * HEIGHT * sizeof(int));
    cudaMalloc(&d_next, WIDTH * HEIGHT * sizeof(int));

    cudaMemcpy(d_current, h_grid.data(), WIDTH * HEIGHT * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((WIDTH + TILE_SIZE - 1) / TILE_SIZE, (HEIGHT + TILE_SIZE - 1) / TILE_SIZE);

    for (int step = 0; step < STEPS; ++step) {
        gameOfLifeShared<<<numBlocks, threadsPerBlock>>>(d_current, d_next, WIDTH, HEIGHT);
        cudaDeviceSynchronize();
        std::swap(d_current, d_next);
    }

    cudaMemcpy(h_grid.data(), d_current, WIDTH * HEIGHT * sizeof(int), cudaMemcpyDeviceToHost);

    // Print a small part of the grid
    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 10; ++x) std::cout << h_grid[y * WIDTH + x] << " ";
        std::cout << std::endl;
    }

    cudaFree(d_current);
    cudaFree(d_next);

    return 0;
}
