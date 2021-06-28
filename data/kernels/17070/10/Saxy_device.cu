#include "includes.h"
__global__ void Saxy_device(float* x, float* y, float* d, float xb, float yb, int n)
{
int i = threadIdx.x;

if (i < n)
d[i] = (x[i] - xb) * (y[i] - yb);

}