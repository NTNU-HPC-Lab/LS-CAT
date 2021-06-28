#include "includes.h"
__global__ void histogram_kernel_optimized(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins) {

// INSERT CODE HERE
extern __shared__ unsigned int bins_s[];

//Shared memory
int thid = threadIdx.x;
while ( thid < num_bins){

bins_s[thid] = 0u;
thid += blockDim.x;
}
__syncthreads();

//Histogram calculation
unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int accumulator = 0;
unsigned int prev_index = 0;

while(element < num_elements){

unsigned int curr_index = input[element];

if(curr_index != prev_index){

atomicAdd(&(bins_s[prev_index]), accumulator);
accumulator = 1;
prev_index = curr_index;

}

else{
accumulator++;
}
element += blockDim.x * gridDim.x;
}
if(accumulator > 0){
atomicAdd(&(bins_s[prev_index]), accumulator);
}
__syncthreads();

//Global memory
thid = threadIdx.x;
while(thid < num_bins){

atomicAdd(&(bins[thid]), bins_s[thid]);
thid += blockDim.x;
}

}