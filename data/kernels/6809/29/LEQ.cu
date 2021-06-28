#include "includes.h"
__global__ void LEQ(float * xf, bool * xb, size_t idxf, size_t idxb, size_t N)
{
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
{
xb[idxb*N+i] = xf[(idxf-1)*N+i] <= xf[(idxf-2)*N+i];
}
return;
}