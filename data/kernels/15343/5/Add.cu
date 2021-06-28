#include "includes.h"
__global__ void Add(float *a, float *b, float *c)
{
int Id = threadIdx.x + blockDim.x * blockIdx.x;
if (Id < N) {
a[Id] = threadIdx.x;
b[Id] = blockIdx.x;
c[Id] = Id;
}
}