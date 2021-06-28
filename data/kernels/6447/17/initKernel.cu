#include "includes.h"
__global__ void initKernel(double* data, int count, double val) {
int ti = blockDim.x * blockIdx.x + threadIdx.x;

if (ti < count) {
data[ti] = val;
}
}