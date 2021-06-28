#include "includes.h"
__global__ void hardTanh_grad(float* in_x, float* in_d, float* out_d, float min_val, float max_val, int size, bool inplace) {
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int stride = gridDim.x * blockDim.x;
for (int i = tid; i < size; i += stride) {
if (inplace) {
if (in_x[i] < min_val || in_x[i] > max_val) in_d[i] = 0;
} else {
if (in_x[i] >= min_val && in_x[i] <= max_val) in_d[i] += out_d[i];
}
}
}