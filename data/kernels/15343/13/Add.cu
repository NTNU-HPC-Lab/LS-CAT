#include "includes.h"
__global__ void Add(float *a, float *b, float *c)
{
int Id = threadIdx.x + blockDim.x * blockIdx.x;
printf("(%d, %d, %d) ", threadIdx.x, blockDim.x, blockIdx.x);
printf("hilo: %d, ", Id);
//solo trabajan los N hilos
if (Id < N) {
c[Id] = a[Id] * b[Id];
}
}