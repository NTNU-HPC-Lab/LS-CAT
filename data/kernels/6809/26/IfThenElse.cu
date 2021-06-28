#include "includes.h"
__global__ void IfThenElse(bool * b, float * x, size_t idxb, size_t idxf, size_t N)
{
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
{
//printf("From IfThenElse %d, %f, %f\n", b[(idxb-1)*N+i], x[(idxf-1)*N+i], x[(idxf-2)*N+i]);
if (b[(idxb-1)*N+i])
x[(idxf-2)*N+i] = x[(idxf-1)*N+i];
//printf("After IfThenElse %f\n", x[(idxf-2)*N+i]);

}
return;
}