#include "includes.h"
__global__ void windowBlackman(float* idata, int length)
{
int tidx = threadIdx.x + blockIdx.x*blockDim.x;
if (tidx < length)
{
idata[tidx] = 0.74 / 2 * -0.5 * cos(2 * PI_F*tidx / (length - 1)) + 0.16 / 2 * sin(4 * PI_F*tidx / (length - 1));
}
}