#include "includes.h"
__global__ void dotProductSingle(int* pFeatureList, float* pValuesList, size_t* pSizeOfInstanceList, size_t pSize, size_t pMaxNnz, float* pDevDotProduct) {
int instanceId = blockIdx.x;
int threadId = threadIdx.x;
float __shared__ value[32];
int __shared__ jumpLength;
size_t __shared__ size;


while (instanceId < pSize) {
value[threadIdx.x] = 0;
if (threadIdx.x == 0) {
jumpLength = instanceId * pMaxNnz;
size = pSizeOfInstanceList[instanceId];
}
__syncthreads();
while (threadId < size) {
value[threadIdx.x] += pValuesList[jumpLength + threadId] *  pValuesList[jumpLength + threadId];

threadId += blockDim.x;
}
// reduce
__syncthreads();
int i = blockDim.x/2;
while (i != 0) {
if (threadIdx.x < i) {
value[threadIdx.x] += value[threadIdx.x + i];
}
__syncthreads();
i /= 2;
}

pDevDotProduct[instanceId] = value[0];
instanceId += gridDim.x;
threadId = threadIdx.x;
}
}