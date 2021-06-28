#include "includes.h"

/*
* Read TODO items below
*/




__global__
__global__ void sharedMatmul(float *a, float *b, float *c, int n)
{

__shared__ float A_tile[32][32];
__shared__ float B_tile[32][32];
int width = gridDim.x*blockDim.x;

float acc = 0;

int i = blockIdx.x*32 + threadIdx.x;
int j = blockIdx.y*32 + threadIdx.y;


/* Accumulate C tile by tile. */

for (int tileIdx = 0; tileIdx < gridDim.x ; tileIdx+=1)
{

/* Load one tile of A and one tile of B into shared mem */

A_tile[threadIdx.y][ threadIdx.x] = a[j * width + tileIdx*32+threadIdx.x];
B_tile[threadIdx.y][threadIdx.x] = b[(tileIdx * 32 + threadIdx.y)* width+ i ];

__syncthreads();

/* Accumulate one tile of C from tiles of A and B in shared mem */

for (int k = 0 ;k < 32; k++)
{
acc += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
}

__syncthreads();

}

c[j * width + i ] = acc;

}