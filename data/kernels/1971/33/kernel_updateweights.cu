#include "includes.h"
__global__ void kernel_updateweights(int N, double *wt, double *x, double *q, double nu){
unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
/* make sure to use only N threads */
if (tid<N) {
wt[tid]=((nu+1.0)/(nu+x[tid]*x[tid]));
q[tid]=wt[tid]-log(wt[tid]); /* so that its +ve */
}
}