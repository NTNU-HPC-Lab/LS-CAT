#include "includes.h"
__global__ void kernelGradf(const float *d_x, float *d_grad)
{
const float x0 = d_x[0];
const float x1 = d_x[1];

// df/dx0 = -2 (1-x0) - 400 (x1-x0^2) x0
// df/dx1 = 200 (x1 - x0^2)

d_grad[0] = -2.0f * (1.0f - x0) - 400.0f * x0 * (x1 - x0*x0);
d_grad[1] = 200.0f * (x1 - x0*x0);
}