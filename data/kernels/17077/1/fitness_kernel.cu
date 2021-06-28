#include "includes.h"
__global__ void fitness_kernel(int* chromosome, int* collision) {
/*unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
unsigned int stride = blockDim.x * gridDim.x;*/
unsigned int tid = threadIdx.x;
unsigned int bid = blockIdx.x;
int temp = chromosome[bid];
int d = 0;
extern __shared__ int cache[]; // to use the thread-block shared memory
cache[tid] = 0;
if (tid < bid) {
d = abs(temp - chromosome[tid]);
if ((d == 0) || (d == (bid - tid))) {
cache[tid] = 1;
}
else {
cache[tid] = 0;
}
}

__syncthreads();

//Reduction
unsigned int i = blockDim.x / 2;
while (i >0) {
if (tid < i) {
cache[tid] += cache[tid + i];
}
__syncthreads();
i /= 2;
}

if (tid == 0) {
atomicAdd(collision, cache[0]);
}

/*while (index < n) {
temp = chromosome[index];
index += stride;
}*/
}