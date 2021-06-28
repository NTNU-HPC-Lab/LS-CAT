#include "includes.h"
/* This file is copied from https://github.com/jzbonter/mc-cnn */
extern "C" {
}



#define TB 128

#define DISP_MAX 256

__global__ void remove_occluded(float *y, int size, int size3)
{
int id = blockIdx.x * blockDim.x + threadIdx.x;
if (id < size) {
int x = id % size3;
for (int i = 1; x + i < size3; i++) {
if (i - y[id + i] < -y[id]) {
y[id] = 0;
break;
}
}
}
}