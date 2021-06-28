#include "includes.h"
extern "C"
{
}
__global__ void vsquare(const double *a, double *c)
{
int i = threadIdx.x + blockIdx.x * blockDim.x;
c[i] = a[i] * a[i];
}