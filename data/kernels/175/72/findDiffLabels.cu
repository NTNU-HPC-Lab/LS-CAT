#include "includes.h"
__global__ void findDiffLabels(float* devDiff, int diffPitchInFloats, int nPoints, int nClusters, int* devClusters, int* devChanges) {
int x = blockDim.x * blockIdx.x + threadIdx.x;
__shared__ unsigned int localChanges;
if (x < nPoints) {
int index = x;
float minDistance = 10000000;
int minCluster = -1;
for(int cluster = 0; cluster < nClusters; cluster++) {
float clusterDistance = devDiff[index];
if (clusterDistance < minDistance) {
minDistance = clusterDistance;
minCluster = cluster;
}
index += diffPitchInFloats;
}
int previousCluster = devClusters[x];
devClusters[x] = minCluster;
if (minCluster != previousCluster) {
atomicInc(&localChanges, 10000000);
}
}
__syncthreads();
if (threadIdx.x == 0) {
atomicAdd(devChanges, localChanges);
}
}