#include "includes.h"
__global__ void kernel_setweights_fl(int N, float *wt, float alpha){
unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
/* make sure to use only M threads */
if (tid<N) {
wt[tid]=alpha;
}
}