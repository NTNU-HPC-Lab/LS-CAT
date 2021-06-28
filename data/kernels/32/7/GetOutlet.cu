#include "includes.h"
__global__ void GetOutlet(double *h, double *houtlet, double *u, double *uout, double *v, double *vout, int M, int N, int t) {
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int ind = 2;
while (tid < M) {
houtlet[t*M+tid] = h[(tid+1)*N-ind];
vout[t*M+tid] = v[(tid+1)*N-ind];
uout[t*M+tid] = u[(tid+1)*N-ind];
tid += blockDim.x * gridDim.x;
}
}