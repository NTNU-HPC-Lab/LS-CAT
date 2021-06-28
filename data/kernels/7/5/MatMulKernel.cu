#include "includes.h"
__global__ void MatMulKernel( float *C, float *A, float *B, int Aheight, int Awidth, int Bwidth ) {
float result = 0;
int elementNum = blockIdx.x * blockDim.x + threadIdx.x;
if( elementNum > Aheight * Bwidth ) {
return;
}
int row = elementNum / Bwidth;
int col = elementNum % Bwidth;
for( int e = 0; e < Awidth; e++ ) {
result += A[row * Awidth + e] * B[e * Bwidth + col];
}
C[row * Bwidth + col] = result;
}