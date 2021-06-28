#include "includes.h"
__global__ void MatrixMulKernel(int *d_x, int *d_y, int *d_z, int Block_Width, int M , int N) {

int row = blockIdx.y*blockDim.y+ threadIdx.y;
int col = blockIdx.x*blockDim.x+ threadIdx.x;

int kernelSum = 0;
if ((row<N) && (col<N)) {
for (int i = 0; i < Block_Width ; ++i) {
kernelSum+=d_x[col * Block_Width + i] * d_y[i * Block_Width + row];
}
}
d_z[row * Block_Width +col] = kernelSum;
}