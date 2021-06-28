#include "includes.h"
__global__ void kernel_updateweights_fl(int N, float *wt, float *x, float *q, float nu){
unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
/* make sure to use only M threads */
if (tid<N) {
wt[tid]=((nu+1.0f)/(nu+x[tid]*x[tid]));
q[tid]=wt[tid]-logf(wt[tid]); /* so that its +ve */
}
}