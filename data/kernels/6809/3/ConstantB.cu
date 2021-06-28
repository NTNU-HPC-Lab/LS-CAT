#include "includes.h"
__global__ void ConstantB(bool * x, bool value, size_t idx, size_t N)
{
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
{
x[(idx)*N + i] = value;
}
return;
}