#include "includes.h"
/* Start Header
***************************************************************** /
/*!
\file knn-kernel.cu
\author Koh Wen Lin
\brief
Contains the implementation for kmeans clustering on the gpu.
*/
/* End Header
*******************************************************************/
#define KMEAN_BLOCK_SIZE 32
#define KMEAN_BLOCK_SIZE_1D KMEAN_BLOCK_SIZE * KMEAN_BLOCK_SIZE


__global__ void Mean(float* dIn, unsigned n, unsigned d, int* dGroupIn, float* dMeanIn, unsigned k, int* count)
{
// Each thread block to perform its own summation internally(Reduction), then, each thread block will add its result into global counter and sum
extern __shared__ float sDataSumGroupCount[]; // Dynamic allocated shared memory enough to store block-size amount of data and sum of cluster, group and count.

float* sData = sDataSumGroupCount;
float* sSum = sData + KMEAN_BLOCK_SIZE_1D * d;
int* sGroup = (int*)&sDataSumGroupCount[(k + KMEAN_BLOCK_SIZE_1D) * d];
int* sCount = sGroup + KMEAN_BLOCK_SIZE_1D;

const int tx = threadIdx.x;
int tid = blockIdx.x * blockDim.x + tx;

if(tid >= n)
return;

// Clear shared memory
if(tx < k)
{
for(int i = 0; i < d; ++i)
sSum[tx * d + i] = dMeanIn[tx * d + i];
sCount[tx] = count[tx] = 0.0f;
}

// Each thread perform 1 global load for all its feature and its group index
memcpy(&sData[tx * d], &dIn[tid * d], d * sizeof(float));
sGroup[tx] = dGroupIn[tid];

// Clear old mean
memset(dMeanIn, 0, k * d * sizeof(float));

// Ensure all data relavant to block is loaded
__syncthreads();

int clusterId = sGroup[tx];

for(int i = 0; i < d; ++i)
atomicAdd(&sSum[clusterId * d + i], sData[tx * d + i]);
atomicAdd(&sCount[clusterId], 1);

__syncthreads();

if(tx == 0)
{
for(int i = 0; i < k * d; ++i)
atomicAdd(&dMeanIn[i], sSum[i]);

for(int i = 0; i < k; ++i)
atomicAdd(&count[i], sCount[i]);
}
}