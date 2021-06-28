#include "includes.h"
__global__ void sphereTransform(float *data, const unsigned int N)
{
unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
if (idx < N)
{
data[idx] = data[idx] * 360.0f - 180.0f;
data[idx + N] = acosf(2.0f * data[idx + N] - 1.0f);
}
}