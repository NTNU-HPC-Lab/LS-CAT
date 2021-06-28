#include "includes.h"
__global__ void matrixAddPitch (int *a, int *b, int*c, int pitch) {

int idx = threadIdx.x + blockIdx.x * blockDim.x;
int idy = threadIdx.y + blockIdx.y * blockDim.y;
if (idx > pitch || idy > HEIGHT) return;

c[idy * pitch + idx] = a[idy * pitch + idx] + b[idy * pitch + idx];
}