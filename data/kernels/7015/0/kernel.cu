#include "includes.h"

namespace ann {








// CUDA2





}

__global__ void kernel(int n, float *arr){

volatile int idx = threadIdx.x + blockDim.x*blockIdx.x;
if(idx >= n) return;

arr[idx] *= 2.0f;
}