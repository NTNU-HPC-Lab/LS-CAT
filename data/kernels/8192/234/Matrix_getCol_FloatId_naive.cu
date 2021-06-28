#include "includes.h"
__global__ void Matrix_getCol_FloatId_naive(const float * A , int Acount, int Acols, float * out0 , int out0count, int out0cols, float col_id)
{
int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
if (id<Acount/Acols)
{
out0[id] = A[id*Acols + (int)col_id];
}
}