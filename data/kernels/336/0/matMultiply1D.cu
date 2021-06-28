#include "includes.h"

//#define NDEBUG


const static float eps = 1e-6;
const static size_t blocSize = 8;
const static size_t size = 1024;



__global__ void matMultiply1D(float* matA, float* matB, float* Dest, int dimensions)
{
int i = threadIdx.x + blockIdx.x*blockDim.x;
if (i < dimensions)
{
float vectA[2048];
for (unsigned k = 0; k != dimensions; ++k)
{
vectA[k] = matB[i*dimensions + k];
}
for (unsigned j = 0; j != dimensions; ++j)
{
float res = 0.0f;
for (unsigned k = 0; k != dimensions; ++k)
{
res += vectA[k] * matB[k*dimensions + j];
}
Dest[i*dimensions + j] = res;
}
}
}