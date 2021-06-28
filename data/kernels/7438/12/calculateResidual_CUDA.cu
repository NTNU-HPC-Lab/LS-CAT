#include "includes.h"
__global__ void calculateResidual_CUDA(float *a, float *b, float *c) {
__shared__ float se[1024];

int tid=threadIdx.x;
int bid=blockIdx.x;
int n=blockDim.x;
//   Calculate
se[tid]=fabsf(a[tid+bid*n]-b[tid+bid*n]);
__syncthreads();

//   Reducto
int numActiveThreads=n/2;
while(numActiveThreads>0) {
if(tid<numActiveThreads) {
se[tid]=se[tid]+se[tid+numActiveThreads];
}
numActiveThreads=numActiveThreads/2;
__syncthreads();
}


if(tid==0) {
atomicAdd(c,se[0]);
}
}