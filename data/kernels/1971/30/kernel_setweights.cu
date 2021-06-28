#include "includes.h"
__global__ void kernel_setweights(int N, double *wt, double alpha){
unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
/* make sure to use only N threads */
if (tid<N) {
wt[tid]=alpha;
}
}