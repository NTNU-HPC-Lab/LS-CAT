#include "includes.h"
__global__ void finishCentroids(int* centroidMass, unsigned int* centroidCount, float* centroids) {
int centroidNumber = blockIdx.y * blockDim.y + threadIdx.y;
int dimensionNumber = blockIdx.x * blockDim.x + threadIdx.x;
if ((centroidNumber < 64) && (dimensionNumber < 34)) {
float totalCount = (float)centroidCount[centroidNumber];
float mass = (float)centroidMass[dimensionNumber * 64 + centroidNumber];
centroids[dimensionNumber * 64 + centroidNumber] = mass / ((float)INTCONFACTOR * totalCount);
}
}