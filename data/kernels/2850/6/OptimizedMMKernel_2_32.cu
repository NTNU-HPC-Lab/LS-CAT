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
__global__ void OptimizedMMKernel_2_32(float *a, float *b, float *c, int size)
{
// Create shared matrices for rows of A and columns of B
__shared__ float sharedA[32][32];
__shared__ float sharedB[32][32];

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
sharedA[ty][tx] = a[(y * size) + (i * 32) + tx];
sharedB[ty][tx] = b[(i * 32 * size) + (ty * size) + x];

// Wait for all threads to load each section of the shared matrix
__syncthreads();

sum += sharedA[ty][0] * sharedB[0][tx];
sum += sharedA[ty][1] * sharedB[1][tx];
sum += sharedA[ty][2] * sharedB[2][tx];
sum += sharedA[ty][3] * sharedB[3][tx];
sum += sharedA[ty][4] * sharedB[4][tx];
sum += sharedA[ty][5] * sharedB[5][tx];
sum += sharedA[ty][6] * sharedB[6][tx];
sum += sharedA[ty][7] * sharedB[7][tx];
sum += sharedA[ty][8] * sharedB[8][tx];
sum += sharedA[ty][9] * sharedB[9][tx];
sum += sharedA[ty][10] * sharedB[10][tx];
sum += sharedA[ty][11] * sharedB[11][tx];
sum += sharedA[ty][12] * sharedB[12][tx];
sum += sharedA[ty][13] * sharedB[13][tx];
sum += sharedA[ty][14] * sharedB[14][tx];
sum += sharedA[ty][15] * sharedB[15][tx];
sum += sharedA[ty][16] * sharedB[16][tx];
sum += sharedA[ty][17] * sharedB[17][tx];
sum += sharedA[ty][18] * sharedB[18][tx];
sum += sharedA[ty][19] * sharedB[19][tx];
sum += sharedA[ty][20] * sharedB[20][tx];
sum += sharedA[ty][21] * sharedB[21][tx];
sum += sharedA[ty][22] * sharedB[22][tx];
sum += sharedA[ty][23] * sharedB[23][tx];
sum += sharedA[ty][24] * sharedB[24][tx];
sum += sharedA[ty][25] * sharedB[25][tx];
sum += sharedA[ty][26] * sharedB[26][tx];
sum += sharedA[ty][27] * sharedB[27][tx];
sum += sharedA[ty][28] * sharedB[28][tx];
sum += sharedA[ty][29] * sharedB[29][tx];
sum += sharedA[ty][30] * sharedB[30][tx];
sum += sharedA[ty][31] * sharedB[31][tx];

// Wait for all threads to compute their partial sum from the shared matrices before loading the next
__syncthreads();
}

// Store the full sum as the result
c[y * size + x] = sum;
}