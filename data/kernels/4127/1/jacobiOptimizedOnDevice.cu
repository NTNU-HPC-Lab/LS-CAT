#include "includes.h"

static char* program_name;

// Usage
__global__ void jacobiOptimizedOnDevice(float* x_next, float* A, float* x_now, float* b, int Ni, int Nj)
{
// Optimization step 1: tiling
int idx = blockIdx.x*blockDim.x + threadIdx.x;

if (idx < Ni)
{
float sigma = 0.0;

// Optimization step 2: store index in register
// Multiplication is not executed in every iteration.
int idx_Ai = idx*Nj;

// Tried to use prefetching, but then the result is terribly wrong and I don't know why..
/*
float curr_A = A[idx_Ai];
float nxt_A;
//printf("idx=%d\n",idx);
for (int j=0; j<Nj-1; j++)
{
if (idx != j)
nxt_A = A[idx_Ai + j + 1];
sigma += curr_A * x_now[j];
//sigma += A[idx_Ai + j] * x_now[j];
curr_A = nxt_A;
//printf("curr_A=%f\n",curr_A);
}
if (idx != Nj-1)
sigma += nxt_A * x_now[Nj-1];
x_next[idx] = (b[idx] - sigma) / A[idx_Ai + idx];
*/

for (int j=0; j<Nj; j++)
if (idx != j)
sigma += A[idx_Ai + j] * x_now[j];

// Tried to use loop-ennrolling, but also here this gives a wrong result..
/*
for (int j=0; j<Nj/4; j+=4)
{
if (idx != j)
{
sigma += A[idx_Ai + j] * x_now[j];
}
if (idx != j+1)
{
sigma += A[idx_Ai + j+1] * x_now[j+1];
}
if (idx != j+2)
{
sigma += A[idx_Ai + j+2] * x_now[j+2];
}
if (idx != j+3)
{
sigma += A[idx_Ai + j+3] * x_now[j+3];
}
}*/

x_next[idx] = (b[idx] - sigma) / A[idx_Ai + idx];
}
}