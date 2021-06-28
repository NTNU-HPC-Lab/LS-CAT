#include "includes.h"

//#define NDEBUG


const static float eps = 1e-6;
const static size_t blocSize = 8;
const static size_t size = 1024;



__global__ void matMultiply2D(float* matA, float* matB, float* Dest, int dimensions)
{
int ix = threadIdx.x + blockIdx.x*blockDim.x;
int iy = threadIdx.y + blockIdx.y*blockDim.y;

if (ix < dimensions&&iy < dimensions)
{
float res = 0.0f;
for (unsigned k = 0; k != dimensions; ++k)
{
res += matA[ix*dimensions + k] * matB[k*dimensions + iy];
}
Dest[ix*dimensions + iy] = res;
}
}