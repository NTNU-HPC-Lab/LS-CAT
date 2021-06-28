#include "includes.h"
__global__ void saxpy_shmem ( float* y, float* x, float a, clock_t * timer_vals)
{
volatile __shared__ float sdata_x0 [COMPUTE_THREADS_PER_CTA];
volatile __shared__ float sdata_y0 [COMPUTE_THREADS_PER_CTA];
int tid = threadIdx.x ;
for (int i=0; i < NUM_ITERS; ++i) {
unsigned int idx = i * COMPUTE_THREADS_PER_CTA * CTA_COUNT + blockIdx.x * COMPUTE_THREADS_PER_CTA + tid;
__syncthreads();
sdata_x0[tid] = x[idx];
sdata_y0[tid] = y[idx];
__syncthreads();
y[idx] = a * sdata_x0[tid] + sdata_y0[tid];
}
}