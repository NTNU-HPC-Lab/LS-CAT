#include "includes.h"
__global__ void vec_sigmoid(float * d, int num) {
const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
if (tid < num) {
if(d[tid] > 10.0) d[tid] = 1.0;
else if(d[tid] < -10.0) d[tid] = 0.0;
else d[tid] = 1.0 / (1.0 + exp(-1.0 * d[tid]));
}
}