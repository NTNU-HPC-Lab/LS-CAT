#include "includes.h"
__global__ void cuda_Pad_Dict(float *PadD, float *D, int nRows_D, int nCols_D, int nFilts, int nRows, int nCols) {
unsigned int Tidx_D = threadIdx.x + blockIdx.x * blockDim.x;
unsigned int Tidy_D = threadIdx.y + blockIdx.y * blockDim.y;

int Dim_D = nRows_D * nFilts;
int i,j;

if ((Tidx_D < nCols_D) && (Tidy_D < nRows_D)) {

for ( i = Tidy_D, j = Tidy_D ; i < Dim_D; i += nRows_D, j += nRows)
PadD[Tidx_D + j * nCols] = D[Tidx_D + i * nCols_D];
}
}