#include "includes.h"
__global__ void If(bool * xb, float * xf, size_t idxf, size_t idxb, size_t N)
{
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
{
/* if (xb[idx-1]) */
/*     xf[idx-1] = xf[idx-1]; */
/* else */
/*     out[i] = 0; */
if (!xb[(idxb-1)*N+i])
xf[(idxf-1)*N+i] = 0;
}
return;
}