#include "includes.h"
__global__ void kernelHelloWorld() {

int thread = threadIdx.x;//local thread number in a block
int block = blockIdx.x;//block number

printf("Hello World from thread %d of block %d!\n", thread, block);
}