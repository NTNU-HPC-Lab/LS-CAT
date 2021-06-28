#include "includes.h"
extern "C" {
}



#define TB 128

#define DISP_MAX 256

__global__ void rho(float *x, int size, float lambda)
{
int id = blockIdx.x * blockDim.x + threadIdx.x;
if (id < size) {
x[id] = 1 - exp(-x[id] / lambda);
}
}