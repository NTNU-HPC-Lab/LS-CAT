#include "includes.h"
__global__ void saxpy_float4s_shmem_doublebuffer ( float* y, float* x, float a, clock_t * timer_vals)
{
volatile __shared__ float sdata_x0_0 [COMPUTE_THREADS_PER_CTA];
volatile __shared__ float sdata_x1_0 [COMPUTE_THREADS_PER_CTA];
volatile __shared__ float sdata_x2_0 [COMPUTE_THREADS_PER_CTA];
volatile __shared__ float sdata_x3_0 [COMPUTE_THREADS_PER_CTA];
volatile __shared__ float sdata_y0_0 [COMPUTE_THREADS_PER_CTA];
volatile __shared__ float sdata_y1_0 [COMPUTE_THREADS_PER_CTA];
volatile __shared__ float sdata_y2_0 [COMPUTE_THREADS_PER_CTA];
volatile __shared__ float sdata_y3_0 [COMPUTE_THREADS_PER_CTA];
volatile __shared__ float sdata_x0_1 [COMPUTE_THREADS_PER_CTA];
volatile __shared__ float sdata_x1_1 [COMPUTE_THREADS_PER_CTA];
volatile __shared__ float sdata_x2_1 [COMPUTE_THREADS_PER_CTA];
volatile __shared__ float sdata_x3_1 [COMPUTE_THREADS_PER_CTA];
volatile __shared__ float sdata_y0_1 [COMPUTE_THREADS_PER_CTA];
volatile __shared__ float sdata_y1_1 [COMPUTE_THREADS_PER_CTA];
volatile __shared__ float sdata_y2_1 [COMPUTE_THREADS_PER_CTA];
volatile __shared__ float sdata_y3_1 [COMPUTE_THREADS_PER_CTA];
int tid = threadIdx.x ;

unsigned int idx0, idx1;
idx0 = blockIdx.x * COMPUTE_THREADS_PER_CTA + tid;
idx1 = COMPUTE_THREADS_PER_CTA * CTA_COUNT + blockIdx.x * COMPUTE_THREADS_PER_CTA + tid;

float4 * x_as_float4 = (float4 *)x;
float4 * y_as_float4 = (float4 *)y;
float4 result_y;

for (int i=0; i < NUM_ITERS/4; i+=2) {
float4 tmp1_x, tmp1_y;

__syncthreads();
tmp1_x = x_as_float4[idx0];
tmp1_y = y_as_float4[idx0];
if (i!=0) {
result_y.x = a * sdata_x0_1[tid] + sdata_y0_1[tid];
result_y.y = a * sdata_x1_1[tid] + sdata_y1_1[tid];
result_y.z = a * sdata_x2_1[tid] + sdata_y2_1[tid];
result_y.w = a * sdata_x3_1[tid] + sdata_y3_1[tid];
y_as_float4[idx1] = result_y;
idx1 += 2 * COMPUTE_THREADS_PER_CTA * CTA_COUNT ;
}
sdata_x0_0[tid] = tmp1_x.x;
sdata_x1_0[tid] = tmp1_x.y;
sdata_x2_0[tid] = tmp1_x.z;
sdata_x3_0[tid] = tmp1_x.w;
sdata_y0_0[tid] = tmp1_y.x;
sdata_y1_0[tid] = tmp1_y.y;
sdata_y2_0[tid] = tmp1_y.z;
sdata_y3_0[tid] = tmp1_y.w;
__syncthreads();
tmp1_x = x_as_float4[idx1];
tmp1_y = y_as_float4[idx1];
result_y.x = a * sdata_x0_0[tid] + sdata_y0_0[tid];
result_y.y = a * sdata_x1_0[tid] + sdata_y1_0[tid];
result_y.z = a * sdata_x2_0[tid] + sdata_y2_0[tid];
result_y.w = a * sdata_x3_0[tid] + sdata_y3_0[tid];
y_as_float4[idx0] = result_y;
idx0 += 2 * COMPUTE_THREADS_PER_CTA * CTA_COUNT ;
sdata_x0_1[tid] = tmp1_x.x;
sdata_x1_1[tid] = tmp1_x.y;
sdata_x2_1[tid] = tmp1_x.z;
sdata_x3_1[tid] = tmp1_x.w;
sdata_y0_1[tid] = tmp1_y.x;
sdata_y1_1[tid] = tmp1_y.y;
sdata_y2_1[tid] = tmp1_y.z;
sdata_y3_1[tid] = tmp1_y.w;
}
__syncthreads();
result_y.x = a * sdata_x0_1[tid] + sdata_y0_1[tid];
result_y.y = a * sdata_x1_1[tid] + sdata_y1_1[tid];
result_y.z = a * sdata_x2_1[tid] + sdata_y2_1[tid];
result_y.w = a * sdata_x3_1[tid] + sdata_y3_1[tid];
y_as_float4[idx1] = result_y;

}