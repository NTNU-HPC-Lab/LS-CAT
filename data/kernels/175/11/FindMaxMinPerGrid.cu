#include "includes.h"
__global__ void FindMaxMinPerGrid(int p_nGridSize, int p_nEigNum, float* p_devMax, float* p_devMin, float* p_devReduceMax, float* p_devReduceMin, int p_nMaxLevel)
{
__shared__ float MaxReduce[XBLOCK*(MAXEIGNUM - 1)];
__shared__ float MinReduce[XBLOCK*(MAXEIGNUM - 1)];

int taskPerTh = (p_nGridSize + XBLOCK - 1)/XBLOCK;
// First Assignment

if (threadIdx.x < p_nGridSize)
{
for (int i = 0; i < p_nEigNum - 1; i++)
{
MaxReduce[i*XBLOCK + threadIdx.x] = p_devMax[threadIdx.x + i * p_nGridSize];
MinReduce[i*XBLOCK + threadIdx.x] = p_devMin[threadIdx.x + i * p_nGridSize];
}
}

// First Reduction
for (int i = 1; i < taskPerTh; i++)
{
int curIndex = threadIdx.x + i * XBLOCK;
if (curIndex < p_nGridSize)
{
for (int j = 0; j < p_nEigNum - 1; j++)
{
if (MaxReduce[j*XBLOCK + threadIdx.x] < p_devMax[curIndex + j * p_nGridSize])
{
MaxReduce[j*XBLOCK + threadIdx.x] = p_devMax[curIndex + j * p_nGridSize];
}
if (MinReduce[j*XBLOCK + threadIdx.x] > p_devMin[curIndex + j * p_nGridSize])
{
MinReduce[j*XBLOCK + threadIdx.x] = p_devMin[curIndex + j * p_nGridSize];
}
}
}
}
__syncthreads();

//The Reductions Thereafter
int mask = 1;
for (int level = 0; level < p_nMaxLevel; level++)
{
if ((threadIdx.x & mask) == 0)
{
int index1 = threadIdx.x;
int index2 = (1 << level) + threadIdx.x;
if (index2 < p_nGridSize)
{
for (int i = 0; i < p_nEigNum - 1; i++)
{
if (MaxReduce[i*XBLOCK + index1] < MaxReduce[i*XBLOCK + index2])
{
MaxReduce[i*XBLOCK + index1] = MaxReduce[i*XBLOCK + index2];
}
if (MinReduce[i*XBLOCK + index1] > MinReduce[i*XBLOCK + index2])
{
MinReduce[i*XBLOCK + index1] = MinReduce[i*XBLOCK + index2];
}
}
}
}
__syncthreads();
mask = (mask<<1)|1;
}

//Write max and min into global memory
if (threadIdx.x == 0)
{
for (int i = 0; i < p_nEigNum - 1; i++)
{
p_devReduceMax[i] = MaxReduce[i*XBLOCK];
p_devReduceMin[i] = MinReduce[i*XBLOCK];
}
}

}