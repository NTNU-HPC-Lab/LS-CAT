#include "includes.h"
__global__ void computeDotProducts(float3* pDotProducts, size_t pSize, int* pCandidates, size_t* pJumpLength, size_t* pCandidateSize, size_t pNumberOfCandidates, int* pFeatureIdsNeighbor, float* pValuesNeighbor, size_t pMaxNnzNeighbor, size_t* pSizeNeighbor, int* pFeatureIdsInstance, float* pValuesInstance, size_t pMaxNnzInstance, size_t* pSizeInstance, float* pPreComputedDotProductsNeighbor, float* pPreComputedDotProductsInstance) {


int instanceCandidates = blockIdx.x;
int round = 0;
int i = 0;
const int threadCount = 32;
__shared__ int instanceCounter;
__shared__ int neighbor;
__shared__ int instance;

__shared__ int featureIdX[threadCount];
__shared__ int featureIdY[threadCount];
__shared__ float value[threadCount];
__shared__ int pStartPosX;
__shared__ int pEndPosX;
__shared__ int pStartPosY;
__shared__ int pEndPosY;

while (instanceCandidates < pNumberOfCandidates) {
if (threadIdx.x == 0) {
neighbor = pCandidates[pJumpLength[instanceCandidates]];
instanceCounter = 0;
}
__syncthreads();
while (instanceCounter < pCandidateSize[neighbor]) {

if (threadIdx.x == 0) {
instance = pCandidates[pJumpLength[instanceCandidates]+instanceCounter];
pStartPosX = neighbor*pMaxNnzNeighbor;
pEndPosX = neighbor*pMaxNnzNeighbor + pSizeNeighbor[neighbor];
pStartPosY = instance*pMaxNnzInstance;
pEndPosY = instance*pMaxNnzInstance + pSizeInstance[instance];
}
value[threadIdx.x] = 0.0;

__syncthreads();

while (pStartPosX < pEndPosX+threadCount - (pEndPosX%threadCount) && pStartPosY < pEndPosY+threadCount - (pEndPosY%threadCount) ) {

featureIdX[threadIdx.x] = pFeatureIdsNeighbor[pStartPosX + threadIdx.x];
featureIdY[threadIdx.x] = pFeatureIdsInstance[pStartPosY + threadIdx.x];

while (round < threadCount) {
if (featureIdX[(threadIdx.x + round) % threadCount] == featureIdY[threadIdx.x]) {
value[threadIdx.x] += pValuesNeighbor[pStartPosX + ((threadIdx.x + round) % threadCount)] * pValuesInstance[pStartPosY + threadIdx.x];
break;
}
++round;
}
__syncthreads();
round = 0;
if (threadIdx.x == 0) {
if (featureIdX[threadCount-1] == featureIdY[threadCount-1]) {
pStartPosY += threadCount;
pStartPosX += threadCount;
} else if (featureIdX[threadCount-1] < featureIdY[threadCount-1]) {
pStartPosX += threadCount;
} else {
pStartPosY += threadCount;
}
}
__syncthreads();
}
__syncthreads();

i = blockDim.x/2;
while (i != 0) {
if (threadIdx.x < i) {
value[threadIdx.x] += value[threadIdx.x + i];
}
__syncthreads();
i /= 2;
}
if (threadIdx.x == 0) {
pDotProducts[pJumpLength[instanceCandidates]+instanceCounter].y = value[0];
pDotProducts[pJumpLength[instanceCandidates]+instanceCounter].x = pPreComputedDotProductsNeighbor[neighbor];
pDotProducts[pJumpLength[instanceCandidates]+instanceCounter].z = pPreComputedDotProductsInstance[instance];
++instanceCounter;
}
__syncthreads();
}
instanceCandidates += gridDim.x;
}
}