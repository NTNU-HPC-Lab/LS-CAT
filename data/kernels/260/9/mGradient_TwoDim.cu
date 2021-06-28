#include "includes.h"
__device__ bool checkBoundary(int blockIdx, int blockDim, int threadIdx){
int x = threadIdx;
int y = blockIdx;
return (x == 0 || x == (blockDim-1) || y == 0 || y == 479);
}
__global__ void mGradient_TwoDim(float *u_dimX, float *u_dimY, float *scalar, float coeffX, float coeffY) {
if(checkBoundary(blockIdx.x, blockDim.x, threadIdx.x)) return;
int Idx = blockIdx.x * blockDim.x + threadIdx.x;
int Left   = Idx - 1;
int Right  = Idx + 1;
int Top    = Idx + blockDim.x;
int Bottom = Idx - blockDim.x;

u_dimX[Idx] -= (scalar[Right] - scalar[Left])*coeffX;
u_dimY[Idx] -= (scalar[Top] - scalar[Bottom])*coeffY;
}