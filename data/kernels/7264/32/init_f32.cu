#include "includes.h"
__global__ void init_f32 (float* vector, float value, int len) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < len) {
vector[idx] = value;
}
}