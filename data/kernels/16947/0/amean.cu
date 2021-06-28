#include "includes.h"
#define N 1024*4
// Device Kernel

//host Function
__global__ void amean(float *A, float *S)
{
//holds intermediates in shared memory reduction
__shared__ int sdata[N];

int tid=threadIdx.x;
int i = blockIdx.x * blockDim.x + threadIdx.x;
sdata[tid]=A[i];
__syncthreads();

for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
if (tid < s) {
sdata[tid] += sdata[tid + s];
}
__syncthreads();
}

if(tid==0)
S[blockIdx.x]=sdata[0];

}