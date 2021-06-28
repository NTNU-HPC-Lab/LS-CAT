#include "includes.h"
/*
#define N 512

#define N 2048
#define THREADS_PER_BLOCK 512

*/
const int THREADS_PER_BLOCK = 32;
const int N = 2048;



__global__ void shared_mult(int *a, int *b, int *c)
{
__shared__ int mem[THREADS_PER_BLOCK];
int pos = threadIdx.x + blockIdx.x * blockDim.x;
mem[threadIdx.x] = a[pos]  * b[pos];

__syncthreads();
c[pos] = mem[threadIdx.x];
}