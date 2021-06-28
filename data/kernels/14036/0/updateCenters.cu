#include "includes.h"
extern "C"
__global__ void updateCenters(float *centers, float *images, int *updates, int noClusters)
{
int gid = blockIdx.x * blockDim.x + threadIdx.x;
int imagesOffset;
int centersIndex=0;

float sum=0;
int index=0;
float weight;
float min;
int minCenterIndex=-1;
int imageSize=784;
float pImage[784];

imagesOffset = gid*imageSize;
for (index=0;index<imageSize;index++){
pImage[index]=images[imagesOffset+index];
}


min=100000000;
for(centersIndex=0;centersIndex<100;centersIndex++)
{
sum = 0;
for(index=0;index<784;index++)
{
weight = centers[centersIndex*imageSize+index]-pImage[index];
sum = sum+weight*weight;
}
if (sum<min)
{
min = sum;
minCenterIndex = centersIndex;
}
}
updates[gid]=minCenterIndex;
}