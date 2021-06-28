#include "includes.h"
__global__ void digitize(float* idat, uint8_t* udat, size_t n)
{
for (
int i = threadIdx.x + blockIdx.x*blockDim.x;
i < n;
i += blockDim.x*gridDim.x)
{
// add an extra 2 here for overhead in case we make it bright
//float tmp = idat[i]/0.02957/2 + 127.5;
// this normalization appears to be more consistent with the VLITE
// digitizers, which have a mean of 128
float tmp = idat[i]/0.02957/2 + 128.5;
if (tmp <= 0)
udat[i] = 0;
else if (tmp >= 255)
udat[i] = 255;
else
udat[i] = (uint8_t) tmp;
}
}