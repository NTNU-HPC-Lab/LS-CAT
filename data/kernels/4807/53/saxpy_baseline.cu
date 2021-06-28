#include "includes.h"
__global__ void saxpy_baseline ( float* y, float* x, float a, clock_t * timer_vals)
{
for (int i=0; i < NUM_ITERS; i++) {
unsigned int idx = i * COMPUTE_THREADS_PER_CTA * CTA_COUNT + blockIdx.x * COMPUTE_THREADS_PER_CTA + threadIdx.x;
y[idx] = a * x[idx] + y[idx];
}
}