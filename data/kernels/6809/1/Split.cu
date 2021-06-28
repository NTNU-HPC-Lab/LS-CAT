#include "includes.h"
__global__ void Split(int * xi, bool * xb, size_t idxi, size_t idxb, size_t N, float threshold)
{
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
{
xb[(idxb)*N+i] = (((float)xi[(idxi-1)*N+i]) == threshold);
}
return;
}