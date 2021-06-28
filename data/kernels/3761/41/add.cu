#include "includes.h"
__global__ void add(int *output, int length, int *n1, int *n2) {
int blockID = blockIdx.x;
int threadID = threadIdx.x;
int blockOffset = blockID * length;

output[blockOffset + threadID] += n1[blockID] + n2[blockID];
}