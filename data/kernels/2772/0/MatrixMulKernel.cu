#include "includes.h"

#define TILE_WIDTH 7

__global__ void MatrixMulKernel(float* Md, float* Nd, float* Pd, int Width)
{
__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

int bx = blockIdx.x; int by = blockIdx.y;
int tx = threadIdx.x; int ty = threadIdx.y;

//Identify the row and column of the Pd element to work on
int Row = by * TILE_WIDTH + ty;
int Col = bx * TILE_WIDTH + tx;

//printf("Row : %d,  Col : %d\n", Row, Col);

float Pvalue = 0;
//Loop over the Md and Nd tiles required to compute the Pd element
for (int m = 0; m < Width / TILE_WIDTH; m++)
{
//Collaborative loading of Md and Nd tiles into shared memory
Mds[ty][tx] = Md[Row * Width + (m * TILE_WIDTH + tx)];
Nds[ty][tx] = Nd[(m * TILE_WIDTH + ty) * Width + Col];
__syncthreads();

for (int k = 0; k < TILE_WIDTH; k++)
Pvalue += Mds[ty][k] * Nds[k][tx];
__syncthreads();
}

Pd[Row * Width + Col] = Pvalue;
}