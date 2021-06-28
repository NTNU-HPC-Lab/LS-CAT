#include "includes.h"
__global__ void kernelF(const float *d_xAx, const float *d_bx, const float *d_c, float *d_y)
{
*d_y = *d_xAx + *d_bx + *d_c;
}