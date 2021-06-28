#include "includes.h"
__global__ void vecAdd(int *xd, float *Ag, float *Bg, float *Cg) {
// this is a kernel, which state the computations the gpu shall do
//int j = threadIdx.x;
int j = blockIdx.x*blockDim.x + threadIdx.x;
*(Cg+j) = *(Ag+j) + *(Bg+j) + (*xd);
}