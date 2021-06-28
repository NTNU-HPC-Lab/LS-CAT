#include "includes.h"
__global__ void Subtract( float * x, size_t idx, size_t N, float W0, float W1)
{
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
{
x[(idx-2)*N+i] = W0*x[(idx-1)*N+i] - W1*x[(idx-2)*N+i];
}
return;
}