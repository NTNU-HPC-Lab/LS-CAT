#include "includes.h"
__global__ void kernel_sqrtweights(int N, double *wt){
unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
/* make sure to use only N threads */
if (tid<N) {
wt[tid]=sqrt(wt[tid]);
}
}