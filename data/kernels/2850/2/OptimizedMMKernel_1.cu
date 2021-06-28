#include "includes.h"

/// Tile size used by the OptimizedMMKernel
#define TILE_SIZE 32

/// Naive matrix multiplication CUDA Kernel

/// Tiled 1D Shared Memory No Unrolling

/// Tiled 2D Shared Memory No Unrolling

/// Tiled 2D Shared Memory With Unrolling (4x4 Tile Size)

/// Tiled 2D Shared Memory With Unrolling (8x8 Tile Size)

/// Tiled 2D Shared Memory With Unrolling (16x16 Tile Size)

/// Tiled 2D Shared Memory With Unrolling (32x32 Tile Size)

/// Prints a matrix out to the stderr stream
__global__ void OptimizedMMKernel_1(float *a, float *b, float *c, int size)
{
// Create shared matrices for rows of A and columns of B
__shared__ float sharedA[TILE_SIZE][TILE_SIZE];
__shared__ float sharedB[TILE_SIZE][TILE_SIZE];

int tx = threadIdx.x;
int ty = threadIdx.y;

int x = blockIdx.x * blockDim.x + tx;
int y = blockIdx.y * blockDim.y + ty;

float sum = 0;

// Divide the matrix up into tiles based on the tile size so each thread
// Can perform its partial sum of the dot product from the shared matrix
int tilesPerGrid = size / blockDim.x;
for (int i = 0; i < tilesPerGrid; i++)
{
// Each thread loads element into A and B
sharedA[ty][tx] = a[(y * size) + (i * TILE_SIZE) + tx];
sharedB[ty][tx] = b[(i * TILE_SIZE * size) + (ty * size) + x];

// Wait for all threads to load each section of the shared matrix
__syncthreads();

for (int j = 0; j < TILE_SIZE; j++)
{
sum += sharedA[ty][j] * sharedB[j][tx];
}

// Wait for all threads to compute their partial sum from the shared matrices before loading the next
__syncthreads();
}

// Store the full sum as the result
c[y * size + x] = sum;
}