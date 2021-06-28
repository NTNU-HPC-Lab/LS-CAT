#include "includes.h"
__global__ void findCentroids(int* responses, int nPixels, int* cluster, int* centroidMass, unsigned int* centroidCount) {
__shared__ int localMasses[32*17];
__shared__ unsigned int localCounts[32];
int pixel = blockDim.x * blockIdx.x + threadIdx.x;
if (threadIdx.x < 32) {
for (int i = 0; i < 17; i++) {
localMasses[32 * i + threadIdx.x] = 0;
}
localCounts[threadIdx.x] = 0;
}
__syncthreads();
if (pixel < nPixels) {
int myCluster = cluster[pixel];
int myIndex = pixel;
for(int filter = 0; filter < 17; filter++) {
int myElement = responses[myIndex];
atomicAdd(localMasses + filter * 32 + myCluster, myElement);
myIndex += nPixels;
}
}
__syncthreads();
if (threadIdx.x < 32) {
for (int filter = 0; filter < 17; filter++) {
atomicAdd(centroidMass + filter * 32 + threadIdx.x, localMasses[threadIdx.x + filter * 32]);
localMasses[threadIdx.x + filter * 32] = 0;
}
}
__syncthreads();
if (pixel < nPixels) {
int myCluster = cluster[pixel];
// yunsup fixed
int myIndex = pixel + nPixels*17;
for(int filter = 0; filter < 17; filter++) {
int myElement = responses[myIndex];
atomicAdd(localMasses + filter * 32 + myCluster, myElement);
myIndex += nPixels;
}
atomicInc(localCounts + myCluster, 100000000);
}
__syncthreads();
if (threadIdx.x < 32) {
for (int filter = 17; filter < 34; filter++) {
atomicAdd(centroidMass + filter * 32 + threadIdx.x, localMasses[threadIdx.x + (filter - 17) * 32]);
}
atomicAdd(centroidCount + threadIdx.x, localCounts[threadIdx.x]);
}
}