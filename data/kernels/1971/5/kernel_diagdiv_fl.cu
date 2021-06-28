#include "includes.h"
__global__ void kernel_diagdiv_fl(int M, float eps, float *y, float *x){
unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
/* make sure to use only M threads */
if (tid<M) {
if (x[tid]>eps) {
y[tid]=y[tid]/x[tid];
} else {
y[tid]=0.0f;
}
}
}