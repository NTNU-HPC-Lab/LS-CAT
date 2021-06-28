#include "includes.h"
__global__ void iReduceSingle2(int *idata, int *single, unsigned int ncols) {
int i;
unsigned int tid = threadIdx.x;
extern __shared__ int sdata[];
unsigned int startPos = blockDim.x + threadIdx.x;
int colsPerThread = ncols/blockDim.x;
int myPart = 0;
for(i=0;i<colsPerThread;i++) {
myPart+=idata[startPos+i];
}
sdata[tid]=myPart;
__syncthreads();

unsigned int s;
for(s=1;s<blockDim.x;s*=2) {
int index = 2*s*tid;
if(index<blockDim.x) {
sdata[index] += sdata[index+s];
}
__syncthreads();
}
if(tid==0)*single=sdata[0];
}