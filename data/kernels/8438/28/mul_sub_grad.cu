#include "includes.h"

using namespace std;
#ifndef MAP_FILE
#define MAP_FILE MAP_SHARED
#endif

__global__ void mul_sub_grad(float* in1_x, float* in1_d, float* in2_x, float* in2_d, float* out, int in1ScalarCount, int in2ScalarCount) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int stride = gridDim.x * blockDim.x;
for (; tid < in1ScalarCount; tid += stride) {
int index = tid % in2ScalarCount;
in1_d[tid] += out[tid] * in2_x[index];
in2_d[tid] = in1_x[tid] * out[tid];  // this is the temp array, need to be reduced!
}
}