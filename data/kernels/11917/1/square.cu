#include "includes.h"
__global__ void square(float * d_out, float * d_in) {
const unsigned int lid = threadIdx.x;
const unsigned int gid = blockIdx.x*blockDim.x + lid;
float f = d_in[gid];
d_out[gid] = f * f;
}