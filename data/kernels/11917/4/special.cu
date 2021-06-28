#include "includes.h"
__global__ void special(float * d_out, float * d_in, int size) {
const unsigned int lid = threadIdx.x;
const unsigned int gid = blockIdx.x*blockDim.x + lid;
if(gid < size) {
float x = d_in[gid];
d_out[gid] = powf(x / (x - 2.3), 3);
}
}