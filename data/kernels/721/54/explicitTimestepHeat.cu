#include "includes.h"
__device__ float explicitLocalStepHeat( float unjpo, float unjmo, float unj, float r)
{
return (1 - 2 * r)*unj + r*unjmo + r * unjpo;
}
__global__ void explicitTimestepHeat( int size, float *d_currentVal, float *d_nextVal, float r )
{
int i = threadIdx.x + blockDim.x * blockIdx.x;
if (i < size)
{
if (i < 2)
{
d_nextVal[i] == 0;
}
else if (i > size - 2)
{
d_nextVal[i] == 0;
}
else
{
d_nextVal[i] = explicitLocalStepHeat(
d_currentVal[i + 1],
d_currentVal[i - 1],
d_currentVal[i],
r);
}
}
}