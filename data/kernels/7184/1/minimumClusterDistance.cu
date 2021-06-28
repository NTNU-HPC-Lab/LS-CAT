#include "includes.h"


#define DIMENSIONS 2
#define GPU_DEVICE_ZERO 0




__global__ void minimumClusterDistance(int threads, double *pointToClusterDistance, int *minimumPointToCluster, int pointsCounter, int clusterCounter)
{
/**
This function puts the point in the right cluster after computing smallest distances.
**/

int leftThreads=pointsCounter % blockDim.x;

if ((blockIdx.x +1 != gridDim.x) || (leftThreads > threadIdx.x))
{
int index=0;
double smallestIndex; //minimum index
double min; //minimum distance
double temp; //temp distance
int pointIndex=threads * blockIdx.x + threadIdx.x;
min = pointToClusterDistance[pointIndex];
int currentIndex;

while(index<clusterCounter)
{

currentIndex=index*pointsCounter;
temp = pointToClusterDistance[pointIndex + currentIndex];
if(temp < min)
{
smallestIndex = index;
min = temp;
}
index++;
}

minimumPointToCluster[pointIndex] = smallestIndex;
}
}