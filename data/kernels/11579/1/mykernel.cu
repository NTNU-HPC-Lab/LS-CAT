#include "includes.h"
__global__ void mykernel(void){
//printf("Inside function\n");
printf("Block id: %d\n",blockIdx.x);
printf("Thread id: %d\n",threadIdx.x);
printf("Global id: %d\n",(threadIdx.x + blockIdx.x*blockDim.x));	//blockDim.x is number of threads in a block
}