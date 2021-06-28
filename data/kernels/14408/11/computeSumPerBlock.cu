#include "includes.h"
__device__ int blockSum(int *b, int size) {
int sum=0, i;
for (i=0; i<size;++i) {
sum += b[i];
}
return sum;
}
__global__ void computeSumPerBlock(int *a, int N) {
//each block has its own sdata_a shared memory area
extern __shared__ int sdata_a[];
int tmp;

//each thread loads 1 element from global to shared memory
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i<N) {
sdata_a[tid] = a[i];
// Make sure we load all values of a to shared memory before
//compute the sum of each subblock.
__syncthreads();

// All blocks execute this in parallel. Note each block has its own
//shared memory sdata_a.
if (tid == 0) {
tmp = blockSum(sdata_a,blockDim.x);
a[i] = tmp;
}
}
}