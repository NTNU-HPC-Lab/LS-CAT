#include "includes.h"
__global__ void HistogramKernel(unsigned int * input, unsigned int size, unsigned int* histogram, unsigned int pass) {
int mid = threadIdx.x + blockIdx.x * blockDim.x;
if (mid < size) {
atomicAdd(&histogram[(input[mid]>>pass) & 0x01], 1);
}
}