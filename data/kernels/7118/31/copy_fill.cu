#include "includes.h"
/* This file is copied from https://github.com/jzbonter/mc-cnn */
extern "C" {
}



#define TB 128

#define DISP_MAX 256

__global__ void copy_fill(float *in, float *out, int size, int in_size2, int in_size3, int out_size2, int out_size3)
{
int id = blockIdx.x * blockDim.x + threadIdx.x;
if (id < size) {
int out_x = id % out_size3;
int out_y = id / out_size3;

int in_x = out_x - (out_size3 - in_size3) / 2;
int in_y = out_y - (out_size2 - in_size2) / 2;

int x = min(in_size3 - 1, max(0, in_x));
int y = min(in_size2 - 1, max(0, in_y));

out[id] = in[y * in_size3 + x];
}
}