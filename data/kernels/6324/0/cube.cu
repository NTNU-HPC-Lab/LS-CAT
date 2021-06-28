#include "includes.h"

// this is how cuda knows that this code is a kernel by calling __global__

__global__ void cube(float * d_out, float * d_in) {
int idx = threadIdx.x ;
float f = d_in[idx];
d_out[idx] = f * f * f;
}