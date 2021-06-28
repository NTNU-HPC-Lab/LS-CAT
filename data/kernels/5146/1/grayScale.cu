#include "includes.h"
__global__ void grayScale(uchar3 *input, uchar3 *output) {
int tid = threadIdx.x + blockIdx.x * blockDim.x;
output[tid].x = (input[tid].x + input[tid].y +
input[tid].z) / 3;
output[tid].z = output[tid].y = output[tid].x;
}