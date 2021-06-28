#include "includes.h"
__global__ void dot( int *a, int *b, int *c ) {

__shared__ int prod[THREADS_PER_BLOCK]; // Shared memory
int index = blockIdx.x * blockDim.x + threadIdx.x;

prod[threadIdx.x] = a[index] * b[index];

__syncthreads();  // Threads synchronization

if( threadIdx.x == 0) {
int par_sum = 0;

for(int i=0; i<THREADS_PER_BLOCK; i++)
par_sum += prod[threadIdx.x]; // Threads reduction

atomicAdd(c,par_sum); // Blocks reduction
}
}