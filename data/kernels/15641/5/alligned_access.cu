#include "includes.h"
__global__ void alligned_access(float* a,int max){
int idx = blockIdx.x*blockDim.x + threadIdx.x;
if (idx >= max) return;
a[idx] = a[idx] + 1;
}