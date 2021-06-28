#include "includes.h"
__global__ void LeftRightBound2D(double *Hs, double *Ztopo, double *K2e, double *K2w, int BC2D, int M, int N) {
int tid = threadIdx.x + blockIdx.x * blockDim.x;
while (tid < M) {
// no-flow BCs
if (BC2D == 0) {
Hs[tid*N] = Hs[tid*N+1];
Hs[(tid+1)*N-1] = Hs[(tid+1)*N-2];

} else {    // Critical depth flow BCs
Hs[tid*N] = hcri + Ztopo[tid*N];
Hs[(tid+1)*N-1] = hcri + Ztopo[(tid+1)*N-1];
}

K2w[tid*N] = K2w[tid*N+1];
K2e[(tid+1)*N-1] = K2e[(tid+1)*N-2];
tid += blockDim.x * gridDim.x;
}
}