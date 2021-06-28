#include "includes.h"
__global__ void Split(float * xf, bool * xb, size_t idxf, size_t idxb, size_t N, float threshold)
{
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
{
xb[(idxb)*N+i] = (xf[(idxf-1)*N+i] < threshold);
}
return;
}