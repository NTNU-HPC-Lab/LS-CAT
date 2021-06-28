#include "includes.h"

#define MAX_SHAREDSIZE	2048


__global__ void LoadStoreViaSharedMemory(int *In, int *Out)
{
#if 1
int LoadStoreSize = MAX_SHAREDSIZE/blockDim.x;
int beginIndex = threadIdx.x * LoadStoreSize;
int endIndex = beginIndex + LoadStoreSize;

// °øÀ¯ ¸Þ¸ð¸® ÇÒ´ç
__shared__ int SharedMemory[MAX_SHAREDSIZE];
int i;

for(i = beginIndex; i < endIndex; i++)
SharedMemory[i] = In[i];

__syncthreads();

for(i = beginIndex; i < endIndex; i++)
Out[i] = SharedMemory[i];

__syncthreads();
#else
__shared__ int SharedMemory[MAX_SHAREDSIZE];

int idx = blockDim.x*blockIdx.x+threadIdx.x;
SharedMemory[idx] = In[idx];
Out[idx] = SharedMemory[idx];
#endif
}