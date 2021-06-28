#include "includes.h"
__global__ void assignInitialClusters(int width, int height, int nPixels, int clusterCount, int* cluster, int filterCount, float* responses, int* intResponses) {
int x = blockDim.x * blockIdx.x + threadIdx.x;
int y = blockDim.y * blockIdx.y + threadIdx.y;
int pixel = y * width + x;
if ((x < width) && (y < height)) {
int xBlock = x / ((width - 1) / 6 + 1);
int yBlock = y / ((height - 1) / 6 + 1);
int assignedCluster = yBlock * 6 + xBlock;

if (assignedCluster >= 32)
{
assignedCluster = 31;
}

cluster[y * width + x] = assignedCluster;
for(int i = 0; i < filterCount; i++) {
int index = pixel + i * nPixels;
int response = (int)(INTCONFACTOR * responses[index]);
intResponses[index] = response;
}
}
}