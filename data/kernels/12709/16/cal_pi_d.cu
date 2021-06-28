#include "includes.h"
__global__ void cal_pi_d(double *sum, int nbin, double step, int nthreads, int nblocks) {
int i;
double x;
int idx = blockIdx.x*blockDim.x+threadIdx.x;  // Sequential thread index across the blocks
for (i=idx; i< nbin; i+=nthreads*nblocks) {
x = (i+0.5)*step;
sum[idx] += 4.0/(1.0+x*x);
}
}