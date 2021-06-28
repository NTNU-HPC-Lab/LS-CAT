#include "includes.h"
__global__ void saxpy(int * a, int * b, int * c)
{
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;

for(int i = tid; i < N; i+=stride){
c[i] = 2 * a[i] + b[i];
}
}