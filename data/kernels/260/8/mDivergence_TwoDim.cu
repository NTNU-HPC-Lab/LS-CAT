#include "includes.h"
__device__ bool checkBoundary(int blockIdx, int blockDim, int threadIdx){
int x = threadIdx;
int y = blockIdx;
return (x == 0 || x == (blockDim-1) || y == 0 || y == 479);
}
__global__ void mDivergence_TwoDim(float *div, float *u_dimX, float *u_dimY, float r_sStep) {
if(checkBoundary(blockIdx.x, blockDim.x, threadIdx.x)) return;
int Idx = blockIdx.x * blockDim.x + threadIdx.x;
int Left   = Idx - 1;
int Right  = Idx + 1;
int Top    = Idx + blockDim.x;
int Bottom = Idx - blockDim.x;

div[Idx] = ((u_dimX[Right]-u_dimX[Left])+(u_dimY[Top]-u_dimY[Bottom]))*r_sStep;
}