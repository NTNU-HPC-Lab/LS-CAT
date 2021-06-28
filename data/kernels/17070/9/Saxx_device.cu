#include "includes.h"
__global__ void Saxx_device(float* x, float* c, float xb, int n)
{
int i = threadIdx.x;
if (i < n)
c[i] = (x[i] - xb) * (x[i] - xb);

}