#include "includes.h"
using namespace std;
#define CUDA_THREAD_NUM 1024
// must be a multiply of 2



void dotProductCPU();
__global__ void dotProductCuda(float *a, float *b, float *c) {
__shared__ float se[CUDA_THREAD_NUM];

// Calculate a.*b
se[threadIdx.x]=a[threadIdx.x+blockIdx.x*CUDA_THREAD_NUM]*b[threadIdx.x+blockIdx.x*CUDA_THREAD_NUM];
__syncthreads();

// Sum Reducto
int numActiveThreads=CUDA_THREAD_NUM/2;
while(numActiveThreads>0) {
if(threadIdx.x<numActiveThreads) {
se[threadIdx.x]=se[threadIdx.x]+se[threadIdx.x+numActiveThreads];
}
numActiveThreads=numActiveThreads/2;
__syncthreads();
}


if(threadIdx.x==0) {
c[blockIdx.x]=se[0];
//printf("BlockId: %d,  ThreadID: %d,  %f \n",blockIdx.x,threadIdx.x,c[blockIdx.x]);
}

return;
}