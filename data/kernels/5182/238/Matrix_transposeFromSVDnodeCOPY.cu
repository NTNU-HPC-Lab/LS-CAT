#include "includes.h"
__global__ void Matrix_transposeFromSVDnodeCOPY(const float* A, int Acount, int Acols, float* out0)
{
int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;

int Arows = Acount/Acols;

int x = id / Arows;
int y = id % Arows;

if (id < Acount)
{
out0[x * Arows + y] = A[y * Acols + x];
}
}