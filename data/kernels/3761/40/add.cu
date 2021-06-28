#include "includes.h"
__global__ void add(int *output, int length, int *n) {
int blockID = blockIdx.x;
int threadID = threadIdx.x;
int blockOffset = blockID * length;

output[blockOffset + threadID] += n[blockID];
}