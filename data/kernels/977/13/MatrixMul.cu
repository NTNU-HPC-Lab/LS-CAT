#include "includes.h"
__global__ void MatrixMul(int *M, int *N, int *P, int width)
{
int bx = blockIdx.x;
int by = blockIdx.y;

int tx = threadIdx.x;
int ty = threadIdx.y;

//int i = by * blockDim.y + ty;
//int j = bx * blockDim.x + tx;

const int tile_size = 16; // tile size

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
As[ty][tx] = M[a + width * ty + tx];
Bs[ty][tx] = N[b + width * ty + tx];
__syncthreads();

//for (int k = 0; k < tile_size; ++k)
//{
//    Csub += As[ty][k] *  Bs[k][tx];
//}
// Loop Unrolling
Csub += As[ty][0] * Bs[0][tx];
Csub += As[ty][1] * Bs[1][tx];
Csub += As[ty][2] * Bs[2][tx];
Csub += As[ty][3] * Bs[3][tx];
Csub += As[ty][4] * Bs[4][tx];
Csub += As[ty][5] * Bs[5][tx];
Csub += As[ty][6] * Bs[6][tx];
Csub += As[ty][7] * Bs[7][tx];
Csub += As[ty][8] * Bs[8][tx];
Csub += As[ty][9] * Bs[9][tx];
Csub += As[ty][10] * Bs[10][tx];
Csub += As[ty][11] * Bs[11][tx];
Csub += As[ty][12] * Bs[12][tx];
Csub += As[ty][13] * Bs[13][tx];
Csub += As[ty][14] * Bs[14][tx];
Csub += As[ty][15] * Bs[15][tx];
__syncthreads();
}

int c = width * tile_size * by + tile_size * bx;
P[c + width * ty + tx] = Csub;
}