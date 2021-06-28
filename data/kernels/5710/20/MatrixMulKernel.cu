#include "includes.h"
__global__ void MatrixMulKernel (float* Md, float* Nd, float* Pd, int ncols) {

int row = blockIdx.y*blockDim.y + threadIdx.y;
int col = blockIdx.x*blockDim.x + threadIdx.x;

// Pvalue is used to store the element of the output matrix
// that is computed by the thread

float Pvalue = 0;
for (int k=0; k < ncols; ++k) {
float Melement = Md[row*ncols+k];
float Nelement = Nd[k*ncols+col];
Pvalue += Melement * Nelement;
}

Pd[row*ncols+col] = Pvalue;
}