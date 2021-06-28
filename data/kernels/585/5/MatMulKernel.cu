#include "includes.h"
__global__ void MatMulKernel(float *Md, float *Nd, float *Pd, int width)
{
// Thread row and column within matrix
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

// Each thread computes one element of P
// by accumulating results into Pvalue
float Pvalue = 0;

// Multiply M and N
for (int k = 0; k < width; ++k) {
float Melement = *(Md + row*width + k);
float Nelement = *(Nd + k*width + col);
Pvalue += Melement * Nelement;
}

// Write Pvalue to device memory
// Each thread writes one element
*(Pd + row*width + col) = Pvalue;
}