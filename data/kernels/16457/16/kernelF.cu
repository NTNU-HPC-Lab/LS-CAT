#include "includes.h"
__global__ void kernelF(const float *d_x, float *d_y)
{
const float &x0 = d_x[0];
const float &x1 = d_x[1];

// f = (1-x0)^2 + 100 (x1-x0^2)^2

const float a = (1.0 - x0);
const float b = (x1 - x0 * x0) ;

*d_y = (a*a) + 100.0f * (b*b);
}