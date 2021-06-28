#include "includes.h"
__global__ void matmul(float* A, float* B, float* C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols) {

float CValue = 0;

int Row = blockIdx.y*16 + threadIdx.y;
int Col = blockIdx.x*16 + threadIdx.x;

for (int k = 0; k < (16 + ACols - 1)/16; k++) {

for (int n = 0; n < 16; ++n)
if ((k*16 + n < ACols && Row < ARows) && (k*16 + n < BRows && Col < BCols))
CValue += A[Row*ACols + k*16 + n] * B[(k*16 + n)*BCols + Col];

}

if (Row < CRows && Col < CCols) C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols)+(blockIdx.x*blockDim.x)+threadIdx.x]=CValue;
}