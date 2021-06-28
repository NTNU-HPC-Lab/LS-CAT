#include "includes.h"
__global__ void SMSV(float* M, float* V, float* R, int* maddr, int* addr, int N) {
int tid = threadIdx.x;
if (tid < N) {
__shared__ int psum[LENGTH];
psum[tid] = 0; // initialize psum with 0s
__syncthreads(); // psum is finished being written to
int numCols = (int)(*addr - (intptr_t)&V[0])/4; // end of SST for vector
for (int i = 0; i <= numCols; i++) { // loop through columns
int vid = (int)V[i+N]; // vector index
int cEnd = (int)(maddr[vid] - (intptr_t)&M[2*N*vid])/4; // end of SST for column
if (tid <= cEnd) {
int mid = (int)M[2*N*vid + tid+N]; // matrix index
psum[mid] += M[2*N*vid + tid] * V[i];
}
__syncthreads();
}
R[tid] = psum[tid];
}
}