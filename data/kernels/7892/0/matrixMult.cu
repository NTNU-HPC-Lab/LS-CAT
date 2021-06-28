#include "includes.h"
/* CUDA API header files*/


extern "C"
__global__ void matrixMult(const double *Md, const double *Nd, double *Pd, int size)
{
int row = blockDim.x * blockIdx.x + threadIdx.x;
int col = blockDim.y * blockIdx.y + threadIdx.y;

if (row < size) {	// Don't do anything to the memory if we're above the size of the matrix
if (col < size) {

double Pvalue = 0;
for (int k = 0; k < size; k++) {
// Elements of 2d-arrays are stored in column-major ordering (i.e. column by column)
// This is a consequence of this code being called in R (where column-major ordering is the norm)
// whereas C usually stores 2d-array in row-major ordering
Pvalue += Md[k*size + row] * Nd[col*size + k];
}
Pd[col*size + row] = Pvalue;

}
}
}