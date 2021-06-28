#include "includes.h"
__global__ void histogram_kernel(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins){

extern __shared__ unsigned int bins_s[];

//Shared Memory
int thid = threadIdx.x;
while(thid < num_bins){

bins_s[thid] = 0u;
thid += blockDim.x;
}
__syncthreads();


//Histogram calculation
unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;

while(element < num_elements){

atomicAdd(&(bins_s[input[element]]), 1);
element += blockDim.x * gridDim.x;
}
__syncthreads();

//Global Memory
thid = threadIdx.x;
while(thid < num_bins){

atomicAdd(&(bins[thid]), bins_s[thid]);
thid += blockDim.x;
}
}