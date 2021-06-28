#include "includes.h"
__global__ void Sign( float * x, size_t idx, size_t N)
{
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
{
float res = x[(idx-1)*N+i];
if (res > 0 )
x[(idx-1)*N+i] = 1.0 ;
else if (res == 0)
x[(idx-1)*N+i] = 0.0;
else
x[(idx-1)*N+i] = -1.0 ;
}
return;
}