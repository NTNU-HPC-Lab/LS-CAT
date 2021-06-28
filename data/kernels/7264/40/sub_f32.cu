#include "includes.h"
__global__ void sub_f32 (float* left_op, float* right_op, float* output, int len) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < len) {
output[idx] = left_op[idx] - right_op[idx];
}
}