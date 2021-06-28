#include "includes.h"
__global__ void SimpleKernel(int N, float* a){
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < N) {
for (int x=0;x<1000;x++)
a[idx] = asin(a[idx]+x);
}
}