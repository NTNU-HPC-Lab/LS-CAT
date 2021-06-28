#include "includes.h"
//============================================================================
// Name        : PoissonEquationJacobiCuda.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================


using namespace std;

const float PI = 4*atan(1);

__global__ void jacobiMethod(float* grid,float* potential, int sizeX,int sizeY,float scale,int noIters,float tolerance){

extern __shared__ float sharedMem[];
/*
Shared memory
1st part is grid
2nd part is initial guess
3rd part is current Solution

*/


// Copying from global to shared memory
int threadIdX = threadIdx.x;
int threadIdY = threadIdx.y;

if (threadIdX == 0 && threadIdY == 0) {
//printf("At Beginning\n");
}


int bOx = blockIdx.x * blockDim.x;
int bOy = blockIdx.y * blockDim.y;

//int totalBlockThreadId = threadIdY*blockDim.x + threadIdX;

//int blockThreadIdx = threadIdX-noIters;
//int blockThreadIdy = threadIdY-noIters;

int effBlockSizeX = blockDim.x + 2 * noIters;
int effBlockSizeY = blockDim.y + 2 * noIters;


int totalSize = sizeX*sizeY;

int sharedMemSize = effBlockSizeX*effBlockSizeY;

for(int i= threadIdX;i<effBlockSizeX;i+= blockDim.x)
for (int j = threadIdY; j < effBlockSizeY; j += blockDim.y) {
int currElemSM = i*effBlockSizeX + j;
int currElemMain = (i - noIters + bOy)*sizeX + (j - noIters + bOx);
if (currElemMain >= 0 && currElemMain < totalSize) {
sharedMem[currElemSM] = grid[currElemMain];
sharedMem[currElemSM + sharedMemSize] = potential[currElemMain];
}
else {
sharedMem[currElemSM] = 0;
sharedMem[currElemSM + sharedMemSize] = 0;
}
sharedMem[currElemSM + 2 * sharedMemSize] = 0;
}
__syncthreads();
if (threadIdX == 0 && threadIdY == 0) {
//printf("Copied to shared memory\n");
}

for(int k=0;k<noIters;k++){
for(int i= threadIdX;i<effBlockSizeX;i+= blockDim.x)
for(int j= threadIdY;j<effBlockSizeY;j+= blockDim.y){
int currPos = i*effBlockSizeX +j+ sharedMemSize*2;
sharedMem[currPos]=0;
if(i>1){
sharedMem[currPos]+=(sharedMem[currPos- effBlockSizeY- sharedMemSize]/4);
}
if(i<effBlockSizeX -1){
sharedMem[currPos]+=(sharedMem[currPos+ effBlockSizeY - sharedMemSize]/4);
}
if(j>1){
sharedMem[currPos]+=(sharedMem[currPos-1- sharedMemSize]/4);
}
if(j<effBlockSizeY-1){
sharedMem[currPos]+=(sharedMem[currPos+1- sharedMemSize]/4);
}
if(i== effBlockSizeX-1||j== effBlockSizeY-1){
//currSolution[currPos]=0;
}else if(currPos - 2 * sharedMemSize>=0){
sharedMem[currPos]+=(scale*scale/4* sharedMem[currPos-2* sharedMemSize]);
}
}
__syncthreads();

for (int i = threadIdX; i<effBlockSizeX; i += blockDim.x)
for (int j = threadIdY; j<effBlockSizeY; j += blockDim.y) {
int currPos = i*effBlockSizeX + j + sharedMemSize * 2;
sharedMem[currPos- sharedMemSize]= sharedMem[currPos];
}
__syncthreads();
}
if (threadIdX == 0 && threadIdY == 0) {
//printf("Done computation\n");
}

for (int i = threadIdX; i<effBlockSizeX; i += blockDim.x)
for (int j = threadIdY; j < effBlockSizeY; j += blockDim.y) {
if (i >= noIters && j >= noIters && i < effBlockSizeX - noIters && j < effBlockSizeX - noIters) {
int currElemSM = i*effBlockSizeX + j;
int currElemMain = (i - noIters + bOy)*sizeX + (j - noIters + bOx);
if (currElemMain > 0 && currElemMain < totalSize) {
potential[currElemMain] = sharedMem[currElemSM + 2 * sharedMemSize];
}

}
}
if (threadIdX == 0 && threadIdY == 0) {
//printf("Copied to memory\n");
}

}