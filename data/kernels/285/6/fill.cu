#include "includes.h"
__global__ void fill(float * w, float val, int size) {
const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
if (tid < size) w[tid] = val;
}