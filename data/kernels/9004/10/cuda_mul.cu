#include "includes.h"
__global__ void cuda_mul(int* A, int* B, int* C, int w)
{
int tid,tx,ty;

//range of tx,ty 0 ~ w
tx = blockDim.x * blockIdx.x + threadIdx.x;
ty = blockDim.y * blockIdx.y + threadIdx.y;
tid = w*ty + tx;

int v = 0;
int a = 0;
int b = 0;


/*
oooo    oxo
xxxx  X oxo
oooo    oxo
oxo
*/

for(int i=0;i< w;i++)
{
a = A[ty * w + i];
b = B[i * w + tx];
v += a+b;
}

C[tid]= v;
}