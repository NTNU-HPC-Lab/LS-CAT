#include "includes.h"
__global__ void sum(int* input, int* sumOut) {
int i = threadIdx.x + blockIdx.x * blockDim.x;
for(int j = 0; j < 100/(blockDim.x*gridDim.x); j++){
if (i < 100){
atomicAdd(sumOut, input[i+(j*blockDim.x*gridDim.x)]);
printf("NUM:%d Thread: %d ||\n",input[i+(j*blockDim.x*gridDim.x)],i);
}
}
__syncthreads();
}