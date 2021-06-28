#include "includes.h"
__global__ void Matrix_MultiplElementWise_naiveOLD(const float * A , int Acount, int Acols, const float * B , int Bcount, int Bcols, float * out0 , int out0count, int out0cols)
{
int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
int id_row,id_col;
if (id<Acount)
{
if (Acount==Bcount) // matrix .* matrix
{
out0[id] = A[id]*B[id];
}
else if (Bcols==1) // matrix .* row vector
{
id_row = id/Acols;
out0[id] = A[id]*B[id_row];
}
else // matrix .* column vector
{
id_col = id%Acols;
out0[id] = A[id]*B[id_col];
}
}
}