#include "includes.h"
__global__ void testKernel(int *s, const int *re){

__shared__ int temp[1];

int i = threadIdx.x;
if (re[i] > -1 && re[i] < temp[0])
temp[0] = re[i];

__syncthreads();

*s = temp[0];
}