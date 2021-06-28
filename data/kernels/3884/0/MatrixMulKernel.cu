#include "includes.h"

/**
* Nathan Dunn
* CS-4370-90 Par. Prog. Many-Core GPUs
* Professor Liu
* 10-24-19
* Tiled Matrix Multiplication
*/

#define N 8 // size of the matrices to be multiplied
#define TILE_WIDTH 4 // size of the tiles

/**
* Computes the matrix multiplication on the CPU
* m - First matrix to be multiplied
* n - Second matrix to be multiplied
* p - Product of m and n
* width - Size of the matrices being operated upon
*/
__global__ void MatrixMulKernel(float* d_M, float* d_N, float* d_P, int Width)
{
__shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
__shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];

int bx = blockIdx.x;  int by = blockIdx.y;
int tx = threadIdx.x; int ty = threadIdx.y;

// Identify the row and column of the Pd element to work on
int Row = by * TILE_WIDTH + ty;
int Col = bx * TILE_WIDTH + tx;

double Pvalue = 0;
// Loop over the Md and Nd tiles required to compute the Pd element
for (int m = 0; m < Width/TILE_WIDTH; ++m){

// Collaborative loading of Md and Nd tiles into shared memory
ds_M[ty][tx] = d_M[Row*Width + m*TILE_WIDTH+tx];
ds_N[ty][tx] = d_N[Col+(m*TILE_WIDTH+ty)*Width];
__syncthreads();
for (int k = 0; k < TILE_WIDTH; ++k)
Pvalue += ds_M[ty][k] * ds_N[k][tx];
__syncthreads();
}
d_P[Row*Width+Col] = Pvalue;
}