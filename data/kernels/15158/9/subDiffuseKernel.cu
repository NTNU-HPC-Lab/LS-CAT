#include "includes.h"
__global__ void subDiffuseKernel(float *data, int x, int y, float pressure)
{
data[NX * x + y] -= pressure;
}