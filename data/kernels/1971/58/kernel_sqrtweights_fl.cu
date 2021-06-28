#include "includes.h"
__global__ void kernel_sqrtweights_fl(int N, float *wt){
unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
/* make sure to use only M threads */
if (tid<N) {
wt[tid]=sqrtf(wt[tid]);
}
}