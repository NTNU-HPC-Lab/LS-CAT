#include "includes.h"
__global__ void Cos( float * x, size_t idx, size_t N, float W0)
{
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
{
x[(idx-1)*N+i] = cos ( W0*x[(idx-1)*N+i] );
}
return;
}