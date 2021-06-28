#include "includes.h"
__global__ void matAdd(int *yd, float *Ag, float *Bg, float *Cg) {
// reverse order of array and gpu idx, to gain speed
int k = blockIdx.x*blockDim.x + threadIdx.x;
int j = blockIdx.y*blockDim.y + threadIdx.y;
*(Cg+j*(*yd)+k) = *(Ag+j*(*yd)+k) + *(Bg+j*(*yd)+k);
}