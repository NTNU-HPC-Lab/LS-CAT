#include "includes.h"
__global__ void cpy(int *a, int *b, int n) {
unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
int sum = 0;
while (i < n) {
int val = b[i];
sum += val;
i += blockDim.x * gridDim.x;
}
atomicAdd(a, sum);
}