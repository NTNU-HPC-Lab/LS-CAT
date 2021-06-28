#include "includes.h"
__global__ void square(float * d_out, float * d_in)
{
int idx = threadIdx.x;
// threadIdx is a C struct (dim3) with 3 members - .x | .y | .z
float f = d_in[idx];
d_out[idx] = f * f;
}