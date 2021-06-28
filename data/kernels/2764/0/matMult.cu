#include "includes.h"

#define BLOCK_SIZE  32
#define N           3200



__global__ void matMult(float* a, float* b, int n, float* c)
{

int   bx = blockIdx.x;
int   by = blockIdx.y;
int   tx = threadIdx.x;
int   ty = threadIdx.y;
float sum = 0.0f;
int   ia = n * BLOCK_SIZE * by + n * ty;
int   ib = BLOCK_SIZE * bx + tx;


for (int k = 0; k < n; k++)
sum += a[ia + k] * b[ib + k * n];

int ic = n * BLOCK_SIZE * by + BLOCK_SIZE * bx;

c[ic + n * ty + tx] = sum;
}