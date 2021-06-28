#include "includes.h"


__global__ void max(int* input, int* maxOut) {
int i = threadIdx.x + blockIdx.x * blockDim.x;
for(int j = 0; j < 100/(blockDim.x*gridDim.x); j++){
if (i < 100){
atomicMax(maxOut, input[i+(j*blockDim.x*gridDim.x)]);
printf("NUM:%d Thread: %d ||\n",input[i+(j*blockDim.x*gridDim.x)],i);
}
}
__syncthreads();
}