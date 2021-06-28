#include "includes.h"
extern "C" {
}



#define TB 128

#define DISP_MAX 256

__global__ void remove_nonvisible(float *y, int size, int size3)
{
int id = blockIdx.x * blockDim.x + threadIdx.x;
if (id < size) {
int x = id % size3;
if (y[id] >= x) {
y[id] = 0;
}
}
}