#include "includes.h"
__global__ void cuda_accumulate_occ(float * device_mapOcc, int numObjs, int numClusters, int clusterStart, int sub_numClusters,  float *device_reduceOcc) {
int objIndex = blockDim.x * blockIdx.x + threadIdx.x;
extern __shared__ float shared_objects[];

if(objIndex < numObjs) {
for(int i = 0;i < sub_numClusters; i++)
//        for(int i = clusterStart + sub_numClusters - 1;i >= clusterStart; i--)
shared_objects[threadIdx.x * sub_numClusters + i] = device_mapOcc[objIndex * numClusters + i + clusterStart];
}
else {
for(int i = 0;i < sub_numClusters; i++)
//for(int i = clusterStart + sub_numClusters - 1;i >= clusterStart; i--)
shared_objects[threadIdx.x * sub_numClusters + i] = 0;
}

__syncthreads();

for(int i = (blockDim.x >> 1); i >= 1; i>>=1) {
if(threadIdx.x < i) {
for(int j = 0;j < sub_numClusters; j++) {
//for(int j = clusterStart + sub_numClusters - 1;j >= clusterStart; j--)
shared_objects[threadIdx.x * sub_numClusters + j] += shared_objects[(threadIdx.x + i) * sub_numClusters + j];
}
}
__syncthreads();
}

if(threadIdx.x == 0) {
for(int i = 0;i < sub_numClusters;i++) {
//        for(int i = clusterStart + sub_numClusters - 1;i >= clusterStart; i--)
device_reduceOcc[blockIdx.x * numClusters + i + clusterStart]  = shared_objects[i];
}
}
}