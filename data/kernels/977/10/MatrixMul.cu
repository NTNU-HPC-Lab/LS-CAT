#include "includes.h"
__global__ void MatrixMul(int *M, int *N, int *P, int width)
{
int bx = blockIdx.x;
int by = blockIdx.y;

int tx = threadIdx.x;
int ty = threadIdx.y;

//int i = by * blockDim.y + ty;
//int j = bx * blockDim.x + tx;

const int tile_size = 16;

__shared__ int As[tile_size][tile_size];
__shared__ int Bs[tile_size][tile_size];

int aBegin = width * tile_size * by;
int aEnd   = aBegin + width - 1;
int aStep  = tile_size;

int bBegin = tile_size * bx;
int bStep  = tile_size * width;

int Csub = 0;
int a, b;

for (a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
{
As[ty][tx] = M[a + width * ty + tx]; // Memory coelescing !
Bs[tx][ty] = N[b + width * ty + tx]; // Memory coelescing !
__syncthreads();

for (int k = 0; k < tile_size; ++k)
{
// For Memory Coalescing :  Bs[k][tx] -> Bs[tx][k]
Csub += As[ty][k] *  Bs[tx][k]; // Bank Conflict on Bs[tx][k]
// It causes Bank Conflict
}
__syncthreads();
}

int c = width * tile_size * by + tile_size * bx;
P[c + width * ty + tx] = Csub;
}