#include "includes.h"


#define DIMENSIONS 2
#define GPU_DEVICE_ZERO 0




__global__ void pointToThreadMove(int pointsCounter, int threadsInsideBlock, double dt, double *pointsInGpu, double *speedArrayInGpu)
{
/**
This function moves the thread with the right velocity readed from the file.
This function puts every point in ONE thread.
**/
int blockDimLeft=pointsCounter % blockDim.x;
if (blockIdx.x != gridDim.x - 1 || blockDimLeft > threadIdx.x)
{
int indexInArray=0;
while(indexInArray < DIMENSIONS)
{
int currentBlock=blockIdx.x * DIMENSIONS * threadsInsideBlock;
int currentThread=threadIdx.x* DIMENSIONS;
int currentGpuPoint = currentBlock + currentThread + indexInArray;
pointsInGpu[currentGpuPoint] += speedArrayInGpu[currentGpuPoint] * dt;
indexInArray++;
}
}
}