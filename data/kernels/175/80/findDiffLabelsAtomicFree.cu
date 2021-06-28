#include "includes.h"
__global__ void findDiffLabelsAtomicFree(float* devDiff, int diffPitchInFloats, int nPoints, int nClusters, int* devClusters, int* devChanges) {


int x = blockDim.x * blockIdx.x + threadIdx.x;
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
//int change=*devChanges;
//change++;
//*devChanges = change;
}
}
}