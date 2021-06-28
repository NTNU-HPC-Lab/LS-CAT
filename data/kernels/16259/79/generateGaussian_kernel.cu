#include "includes.h"
__global__ void generateGaussian_kernel(float* og, float delta, int radius)
{
int x = threadIdx.x - radius;
og[threadIdx.x] = __expf(-(x * x) / (2 * delta * delta));
}