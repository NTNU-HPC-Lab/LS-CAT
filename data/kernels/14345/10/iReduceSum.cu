#include "includes.h"
__global__ void iReduceSum(int *idata, int *odata, unsigned int ncols) {
int i;
unsigned int tid = threadIdx.x;
extern __shared__ int sdata[];
unsigned int startPos = blockDim.x + threadIdx.x;
int colsPerThread = ncols/blockDim.x;
int blockOffset = threadIdx.x *(ncols/blockDim.x);
int myPart = 0;
for(i=0;i<colsPerThread;i++) {
myPart+=idata[blockOffset+startPos+i];
}
sdata[tid]=myPart;
__syncthreads();

unsigned int s;
for(s=1;s<blockDim.x;s*=2){
if(tid%(2*s) == 0){
sdata[tid]+=sdata[tid+s];
}
__syncthreads();
}
if(tid==0)odata[blockIdx.x]=sdata[0];
}