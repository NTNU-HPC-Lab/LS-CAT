#include "includes.h"
__global__ void matrixMul_sharedMemory(float *M, float *N, float *P, int m, int j, int n)
{
__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

int bx = blockIdx.x;
int by = blockIdx.y;
int tx = threadIdx.x;
int ty = threadIdx.y;

int Row = by * TILE_WIDTH + ty;
int Col = bx * TILE_WIDTH + tx;

float pValue = 0;
for(int ph =0; ph < ceil(j/(float)TILE_WIDTH); ph++)
{
if(Row < m && ph * TILE_WIDTH + tx < j)
Mds[ty][tx] = M[Row * j + ph * TILE_WIDTH + tx];   // M[Row][ph * TILE_WIDTH + tx]
if(Col < n && ph * TILE_WIDTH + ty < j)
Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) *n + Col];   // N[ph * TILE_WIDTH + ty][Col]
__syncthreads();

for(int k = 0; k <TILE_WIDTH; k++)
{
if(ph * TILE_WIDTH + k < j)
pValue += Mds[ty][k] * Nds[k][tx];
}

__syncthreads();
}
if(Row < m && Col < n)
P[Row * n + Col] = pValue;             //  整个代码怎么理解呢？ 要有block并行的想法，每个block都有shared memory，
//  这儿我理解是每个block都申请了Tile_width*Tile_width的内存,以block为单位来想这个程序。
}