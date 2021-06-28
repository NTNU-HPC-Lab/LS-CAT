#include "includes.h"
__global__ void prefSumBinTreeCuda(float *a, int n) {
__shared__ float shm[CUDA_THREAD_NUM];
int tid=threadIdx.x;
int dot=2;//depth of tree

if((tid+1)%dot==0) {
shm[tid]=a[tid]+a[tid-1];
}
dot*=2;
__syncthreads();
while(dot<=n)  {
if((tid+1)%dot==0) {
shm[tid]=shm[tid]+shm[tid-dot/2];
}
dot*=2;
__syncthreads();
}
dot/=2;
while(dot>2) {
if((tid+1)%dot==0) {
if((tid+1)/dot!=1) {
shm[tid-dot/2]=shm[tid-dot/2]+shm[tid-dot];
}
}
dot/=2;
__syncthreads();
}

if((tid+1)%2==0) {
a[tid]=shm[tid];
} else if(tid>0) {
a[tid]=a[tid]+shm[tid-1];

}

}