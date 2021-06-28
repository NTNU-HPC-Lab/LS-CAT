#include "includes.h"
__global__ void saxpy_float4s ( float* y, float* x, float a, clock_t * timer_vals)
{
for (int i=0; i < NUM_ITERS/4; i++) {
unsigned int idx = i * COMPUTE_THREADS_PER_CTA * CTA_COUNT + blockIdx.x * COMPUTE_THREADS_PER_CTA + threadIdx.x;

float4 * x_as_float4 = (float4 *)x;
float4 * y_as_float4 = (float4 *)y;

float4 tmp1_x, tmp1_y;
tmp1_x = x_as_float4[idx];
tmp1_y = y_as_float4[idx];

float4 result_y;
result_y.x = a * tmp1_x.x + tmp1_y.x;
result_y.y = a * tmp1_x.y + tmp1_y.y;
result_y.z = a * tmp1_x.z + tmp1_y.z;
result_y.w = a * tmp1_x.w + tmp1_y.w;
y_as_float4[idx] = result_y;
}
}