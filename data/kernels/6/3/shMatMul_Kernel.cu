#include "includes.h"
__global__ void shMatMul_Kernel(int matrixSize, float* matrixA, float* matrixB, float* matrixC)
{
extern __shared__ float sh_Mem[];
int tilewidth = blockDim.x;
float *sh_MatrixA = &(sh_Mem[0]);
float *sh_MatrixB = &(sh_Mem[1*tilewidth*tilewidth]);
//float *sh_MatrixC= &(sh_Mem[2*tilewidth*tilewidth]);

int elementIdx = blockIdx.x * blockDim.x + threadIdx.x; // Col
int elementIdy = blockIdx.y * blockDim.y + threadIdx.y; // Row

int elementId = elementIdy * matrixSize + elementIdx;
float CValue = 0;
if (elementIdx < matrixSize && elementIdy < matrixSize) {
for(int m=0; m < (matrixSize/tilewidth); ++m)
{
sh_MatrixA[tilewidth*threadIdx.y + threadIdx.x] = matrixA[elementIdy*matrixSize + (m*tilewidth+threadIdx.x)];
sh_MatrixB[tilewidth*threadIdx.y + threadIdx.x] = matrixB[elementIdx + (m*tilewidth+threadIdx.y)*matrixSize];
__syncthreads();

for(int k=0; k<tilewidth; ++k)
CValue += sh_MatrixA[tilewidth*threadIdx.y + k] * sh_MatrixB[tilewidth*k + threadIdx.x];
__syncthreads();
}
matrixC[elementId] = CValue;
}
}