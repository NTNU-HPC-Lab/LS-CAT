#include "includes.h"
__global__ void kernel(float *a, int offset)
{
int i = offset + threadIdx.x + blockIdx.x*blockDim.x;
float x = (float)i;
float s = sinf(x);
float c = cosf(x);
a[i] = a[i] + sqrtf(s*s+c*c);
}