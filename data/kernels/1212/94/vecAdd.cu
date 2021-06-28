#include "includes.h"
__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
if (i < len)
out[i] = in1[i] + in2[i];
}