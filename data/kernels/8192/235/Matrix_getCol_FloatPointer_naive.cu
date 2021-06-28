#include "includes.h"
__global__ void Matrix_getCol_FloatPointer_naive(const float * A , int Acount, int Acols, const float * colId , int empty_par1, int empty_par2, float * out0 , int out0count, int out0cols)
{
int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
if (id<Acount/Acols)
{
out0[id] = A[id*Acols + (int)colId[0]];
}
}