#include "includes.h"
__global__ void saxpy_float4s_shmem ( float* y, float* x, float a, clock_t * timer_vals)
{
volatile __shared__ float sdata_x0 [COMPUTE_THREADS_PER_CTA];
volatile __shared__ float sdata_x1 [COMPUTE_THREADS_PER_CTA];
volatile __shared__ float sdata_x2 [COMPUTE_THREADS_PER_CTA];
volatile __shared__ float sdata_x3 [COMPUTE_THREADS_PER_CTA];
volatile __shared__ float sdata_y0 [COMPUTE_THREADS_PER_CTA];
volatile __shared__ float sdata_y1 [COMPUTE_THREADS_PER_CTA];
volatile __shared__ float sdata_y2 [COMPUTE_THREADS_PER_CTA];
volatile __shared__ float sdata_y3 [COMPUTE_THREADS_PER_CTA];
int tid = threadIdx.x ;

for (int i=0; i < NUM_ITERS/4; i++) {
unsigned int idx = i * COMPUTE_THREADS_PER_CTA * CTA_COUNT + blockIdx.x * COMPUTE_THREADS_PER_CTA + threadIdx.x;

__syncthreads();
float4 * x_as_float4 = (float4 *)x;
float4 * y_as_float4 = (float4 *)y;
float4 tmp1_x, tmp1_y;
tmp1_x = x_as_float4[idx];
tmp1_y = y_as_float4[idx];
sdata_x0[tid] = tmp1_x.x;
sdata_x1[tid] = tmp1_x.y;
sdata_x2[tid] = tmp1_x.z;
sdata_x3[tid] = tmp1_x.w;
sdata_y0[tid] = tmp1_y.x;
sdata_y1[tid] = tmp1_y.y;
sdata_y2[tid] = tmp1_y.z;
sdata_y3[tid] = tmp1_y.w;
__syncthreads();

float4 result_y;
result_y.x = a * sdata_x0[tid] + sdata_y0[tid];
result_y.y = a * sdata_x1[tid] + sdata_y1[tid];
result_y.z = a * sdata_x2[tid] + sdata_y2[tid];
result_y.w = a * sdata_x3[tid] + sdata_y3[tid];
y_as_float4[idx] = result_y;
}

}