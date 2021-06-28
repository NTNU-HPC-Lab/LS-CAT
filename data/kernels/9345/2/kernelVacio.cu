#include "includes.h"
__global__ void kernelVacio( void ) {
if (threadIdx.x < 10) {
printf("Data: %s Id Thread: %d Id block : %d Num threads block : %d\n", "helloWorld!", threadIdx.x, blockIdx.x, blockDim.x);
}
}