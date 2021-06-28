#include "includes.h"
__global__ void computePressure_CUDA(float* pressure, float* density, const int num, const float rho0, const float stiff)
{
const unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
if (i >= num) return;
pressure[i] = stiff * (powf((density[i] / rho0), 7) - 1.0f);
//clamp
if (pressure[i] < 0.0f) pressure[i] = 0.0f;
return;
}