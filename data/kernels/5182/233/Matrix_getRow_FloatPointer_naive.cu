#include "includes.h"
__global__ void Matrix_getRow_FloatPointer_naive(const float * A , int Acount, int Acols, const float * rowId , int empty_par1, int empty_par2, float * out0 , int out0count, int out0cols)
{
int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
if (id<Acols)
{
out0[id] = A[id + (int)rowId[0]*Acols];
}
}