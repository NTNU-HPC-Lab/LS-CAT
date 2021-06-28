#include "includes.h"
extern "C" {
}



#define TB 128

#define DISP_MAX 256

__global__ void remove_white(float *x, float *y, int size)
{
int id = blockIdx.x * blockDim.x + threadIdx.x;
if (id < size) {
if (x[id] == 255) {
y[id] = 0;
}
}
}