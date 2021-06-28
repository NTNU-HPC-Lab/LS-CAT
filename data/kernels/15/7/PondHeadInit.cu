#include "includes.h"
__global__ void PondHeadInit(double *ph, int size) {
int tid = threadIdx.x + blockIdx.x * blockDim.x;
while (tid < size) {
ph[tid] = psi_min;
tid += blockDim.x * gridDim.x;
}
}