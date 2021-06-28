#include "includes.h"
__global__ void Add(float * x, size_t idx, size_t N, float W0, float W1)
{
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
{
//printf("Adding %f and %f\n",x[(idx-1)*N + i], x[(idx-2)*N + i]);
//printf("idx = %d, N = %d, i = %d\n", idx, N, i);
//printf("%f %f %f %f %f %f\n", x[0], x[1], x[2], x[3], x[4], x[5]);
x[(idx-2)*N + i] = x[(idx-1)*N + i]*W0 + x[(idx-2)*N + i]*W1;
//printf("on stack %f\n",x[(idx-2)*N + i]);
//printf("%f %f %f %f\n", x[0], x[1], x[2], x[3]);//, x[4], x[5]);
}
return;
}