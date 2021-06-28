#include "includes.h"
__global__ void Not( bool * x, size_t idx, size_t N)
{
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
{
x[(idx-1)*N+i] = ! x[(idx-1)*N+i] ;
}
return;
}