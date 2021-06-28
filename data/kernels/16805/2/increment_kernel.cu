#include "includes.h"
__global__ void increment_kernel(int *g_data, int inc_value) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
g_data[idx] = g_data[idx] + inc_value;
}