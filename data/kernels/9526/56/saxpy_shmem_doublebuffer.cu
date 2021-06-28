#include "includes.h"
__global__ void saxpy_shmem_doublebuffer ( float* y, float* x, float a, clock_t * timer_vals)
{
volatile __shared__ float sdata_x0 [COMPUTE_THREADS_PER_CTA];
volatile __shared__ float sdata_y0 [COMPUTE_THREADS_PER_CTA];
volatile __shared__ float sdata_x1 [COMPUTE_THREADS_PER_CTA];
volatile __shared__ float sdata_y1 [COMPUTE_THREADS_PER_CTA];
int tid = threadIdx.x ;
unsigned int idx0, idx1;
idx0 = blockIdx.x * COMPUTE_THREADS_PER_CTA + tid;
idx1 = COMPUTE_THREADS_PER_CTA * CTA_COUNT + blockIdx.x * COMPUTE_THREADS_PER_CTA + tid;
for (int i=0; i < NUM_ITERS; i+=2) {
__syncthreads();
sdata_x0[tid] = x[idx0];
sdata_y0[tid] = y[idx0];
if (i!=0) {
y[idx1] = a * sdata_x1[tid] + sdata_y1[tid];
idx1 += 2 * COMPUTE_THREADS_PER_CTA * CTA_COUNT ;
}
__syncthreads();
sdata_x1[tid] = x[idx1];
sdata_y1[tid] = y[idx1];
y[idx0] = a * sdata_x0[tid] + sdata_y0[tid];
idx0 += 2 * COMPUTE_THREADS_PER_CTA * CTA_COUNT ;
}
__syncthreads();
y[idx1] = a * sdata_x1[tid] + sdata_y1[tid];
}