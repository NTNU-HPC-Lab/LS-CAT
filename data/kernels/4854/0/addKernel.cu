#include "includes.h"

// GPU¸¦ À§ÇÑ Ä¿³Î ÇÁ·Î±×·¥(NVCC°¡ ÄÄÆÄÀÏÇÔ)

__global__ void addKernel(int* c, const int * a, const int * b)
{
int i = threadIdx.x;
c[i] = a[i] + b[i];
}