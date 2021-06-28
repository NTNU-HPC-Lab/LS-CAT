#include "includes.h"
__global__ void convert_kernel(unsigned int *bins32, uint8_t *bins8, unsigned int num_bins) {

// INSERT CODE HERE
int thid = blockIdx.x * blockDim.x + threadIdx.x;

while (thid < num_bins){

//Use local  register value (avoids copying from global twice)
unsigned int reg_bin = bins32[thid];

if(reg_bin > 255){
bins8[thid] = 255u;
}

else{
bins8[thid] = (uint8_t) reg_bin;
}
thid += blockDim.x * gridDim.x;
}

}