#include "includes.h"
__global__ void copy_kernel(double *save, double *y) {
const int threadID = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
save[threadID] = y[threadID];
save[threadID + 1] = y[threadID + 1];
}