#include "includes.h"
__global__ void windowBlackman2d(float* idata, int length, int height)
{
int tidx = threadIdx.x + blockIdx.x*blockDim.x;
int tidy = threadIdx.y + blockIdx.y*blockDim.y;
if (tidx < length && tidy < height)
{
idata[tidy * length + tidx] = (0.74 / 2 * -0.5 * cos(2 * PI_F*tidy / (height - 1)) + 0.16 / 2 * sin(4 * PI_F*tidy / (height - 1)))
* (0.74 / 2 * -0.5 * cos(2 * PI_F*tidx / (length - 1)) + 0.16 / 2 * sin(4 * PI_F*tidx / (length - 1)));
}
}