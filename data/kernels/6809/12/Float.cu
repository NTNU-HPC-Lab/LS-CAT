#include "includes.h"
__global__ void Float(float * x, bool* y, size_t idxf, size_t idxb, size_t N)
{
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
x[(idxf)*N + i] = float(y[(idxb-1)*N + i]);

return;
}