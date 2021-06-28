#include "includes.h"
__global__ void factorKernel(float *w, int N)
{
int ix  = blockIdx.x * blockDim.x + threadIdx.x;
int idx = ix * 2;
int izx = N + idx;

const float pi = 3.141592653589793238462643383;
float aw = (2.0 * pi) / (float)N;
float arg = aw * (float)ix;

/* Twiddle factors are symmetric along N/2. with change in sign, due to 180 degree phase change */
if (idx < N) {
w[idx] = cos(arg);
w[idx + 1] = sin(arg);
w[izx] = (-1) * w[idx];
w[izx+1] = (-1) * w[idx + 1];
}
}