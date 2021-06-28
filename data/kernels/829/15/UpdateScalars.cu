#include "includes.h"
__global__ void UpdateScalars(float *WHAT , float *WITH , float AMOUNT , float *MASS) {
int idx = threadIdx.x + blockIdx.x * blockDim.x; // this defines the element
WHAT[idx] += AMOUNT*WITH[idx]/MASS[idx];
}