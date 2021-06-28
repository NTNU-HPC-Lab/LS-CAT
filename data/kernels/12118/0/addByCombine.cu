#include "includes.h"



#define N (2048*2048)
#define N_THREADS_PER_BLOCK 512

// Adapt vector addition to use both blocks and threads

__global__ void addByCombine(int *a, int *b, int *c)
{
// use the built-in variable blockDim.x for threads per block
int index = threadIdx.x + blockIdx.x * blockDim.x;
c[index] = a[index] + b[index];
}