#include "includes.h"
__global__ void invierte(float *a, float *b) {
int id = threadIdx.x;
//int id = threadIdx.x + blockDim.x * blockIdx.x;// para n-bloques de 1 hilo

if (id < N)
{
b[id] = a[N-id];
}
}