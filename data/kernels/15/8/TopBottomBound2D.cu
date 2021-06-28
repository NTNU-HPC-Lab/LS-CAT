#include "includes.h"
__global__ void TopBottomBound2D(double *Hs, double *Ztopo, double *K2n, double *K2s, int BC2D, int M, int N) {
int tid = threadIdx.x + blockIdx.x * blockDim.x;

while (tid < N) {
// no-flow BCs
if (BC2D == 0) {
Hs[tid] = Hs[N+tid];
Hs[(M-1)*N+tid] = Hs[(M-2)*N+tid];

} else {    // Critical depth flow BCs
Hs[tid] = hcri + Ztopo[tid];
Hs[(M-1)*N+tid] = hcri + Ztopo[(M-1)*N+tid];
}

K2s[tid] = K2s[N+tid];
K2n[(M-1)*N+tid] = K2n[(M-2)*N+tid];

tid += blockDim.x * gridDim.x;
}
}