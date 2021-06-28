#include "includes.h"
__global__ void sReduceSingle(int *idata,int *single,unsigned int ncols) {
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
for(s=blockDim.x/2;s>0;s>>=1) {
if(tid<s) {
sdata[tid] += sdata[tid+s];
}
__syncthreads();
}
if(tid==0)*single=sdata[0];

}