#include "includes.h"
__global__ void initialize(float* a, float* oA, float* x, float totalSize, int n, int ghosts){
int i = threadIdx.x + blockDim.x*blockIdx.x;
for(int j = 0; blockDim.x*j + i < n + 2*ghosts; j++){
int index = blockDim.x*j + i;
a[index] = 0;
oA[index] = 0;
x[index] = totalSize/n;
}
}