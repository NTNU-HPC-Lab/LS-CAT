#include "includes.h"
__global__ void lumHistExclusiveScan_kernel(unsigned int *d_out, unsigned int *d_in, int numItems)
{
extern __shared__ unsigned int s_exScan[];
int tid = threadIdx.x;

s_exScan[tid] = (tid > 0) ? d_in[tid - 1] : 0;
__syncthreads();

for (int offset = 1; offset <= numItems; offset = offset * 2){
unsigned int temp = s_exScan[tid];
unsigned int neighbor = 0;
if ((tid - offset) >= 0) {
neighbor = s_exScan[tid - offset];
__syncthreads();
s_exScan[tid] = temp + neighbor;
}
__syncthreads();
}
d_out[tid] = s_exScan[tid];
}