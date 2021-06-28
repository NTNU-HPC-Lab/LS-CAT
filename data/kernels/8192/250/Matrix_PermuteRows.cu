#include "includes.h"
__global__ void Matrix_PermuteRows(const float * A , int Acount, int Acols, const float * B , int Bcount, int Bcols, float * out0 , int out0count, int out0cols)
{
int id = blockDim.x*blockIdx.y*gridDim.x	+   blockDim.x*blockIdx.x	  +   threadIdx.x;
int id_row, id_col, id_rowNew;
if (id<Acount)
{
id_row = id/Acols;
id_col = id%Acols;
id_rowNew = B[id_row]*Acols;
out0[id] = A[id_col + id_rowNew];
}
}