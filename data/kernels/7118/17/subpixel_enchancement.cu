#include "includes.h"
/* This file is copied from https://github.com/jzbonter/mc-cnn */
extern "C" {
}



#define TB 128

#define DISP_MAX 256

__global__ void subpixel_enchancement(float *d0, float *c2, float *out, int size, int dim23, int disp_max) {
int id = blockIdx.x * blockDim.x + threadIdx.x;
if (id < size) {
int d = d0[id];
out[id] = d;
if (1 <= d && d < disp_max - 1) {
float cn = c2[(d - 1) * dim23 + id];
float cz = c2[d * dim23 + id];
float cp = c2[(d + 1) * dim23 + id];
float denom = 2 * (cp + cn - 2 * cz);
if (denom > 1e-5) {
out[id] = d - min(1.0, max(-1.0, (cp - cn) / denom));
}
}
}
}