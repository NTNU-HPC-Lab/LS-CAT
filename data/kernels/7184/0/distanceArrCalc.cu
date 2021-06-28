#include "includes.h"


#define DIMENSIONS 2
#define GPU_DEVICE_ZERO 0




__global__ void distanceArrCalc(int pointsCounter, int threads, double *distanceFromPointToCluster, double *pointsInGpu, double *clustersInGpu)
{
/**
This Function computes distances. Every index is a point. Every value inside an index is a distance.
**/
double distanceX = 0;
double distanceY= 0;

int threadsLeft=pointsCounter % blockDim.x;
if ((threadsLeft > threadIdx.x) || (blockIdx.x+1 != gridDim.x))
{
int offsetPointIndex=(blockIdx.x * threads + threadIdx.x)*DIMENSIONS;
int offsetClusterIndexForPoint=threadIdx.y * DIMENSIONS;

//calc X
double a=pointsInGpu[offsetPointIndex];
double b= clustersInGpu[offsetClusterIndexForPoint];
distanceX = (a - b);
distanceX*=distanceX;

//calc Y
a=pointsInGpu[offsetPointIndex+1];
b= clustersInGpu[offsetClusterIndexForPoint+1];
distanceY =(a - b);
distanceY*=distanceY;


double totalDistance=sqrt(distanceY+distanceX);
int currentPointIndexY = pointsCounter*threadIdx.y;
int currentPointIndexX=(blockIdx.x * threads + threadIdx.x);
int pointIndex=currentPointIndexY+currentPointIndexX;

distanceFromPointToCluster[pointIndex] = totalDistance;
}
}