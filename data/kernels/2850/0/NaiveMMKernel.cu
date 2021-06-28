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
__global__ void NaiveMMKernel(float *a, float *b, float *c, int size)
{
int xOut = blockDim.x * blockIdx.x + threadIdx.x;
int yOut = blockDim.y * blockIdx.y + threadIdx.y;

float outValue = 0;
for (int i = 0; i < size; i++)
{
// Row of a mulitplied by the column of b
float prod = a[yOut * size + i] * b[i * size + xOut];
outValue += prod;
}

// Store sum of dot products in C matrix
c[yOut * size + xOut] = outValue;
}