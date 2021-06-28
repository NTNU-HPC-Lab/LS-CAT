#include "includes.h"
__global__ void square(float* d_out, float* d_in)
{
int idx = threadIdx.x;   // here depends on the <<<block, threadPerBlock>>>,  build-in variable: threadIdx
float f = d_in[idx];
d_out[idx] = f * f;
}