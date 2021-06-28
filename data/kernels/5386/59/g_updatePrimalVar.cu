#include "includes.h"
__device__ inline float d_square_prox(float x0, float c, float f, float tau) {
return (x0 + 2.f * tau * c * f) / (1.f + 2.f * tau * c * c);
}
__device__ void d_calcDivergence(const float *v1, const float *v2, float &divv, size_t width, size_t height, size_t c, const bool *mask) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
size_t i_mask = x + y * width;
size_t i      = x + y*width + c * width * height;

float v1x = 0.f, v2y = 0.f;
if (x>0 && mask[i_mask] && mask[i_mask-1]) v1x = v1[i] - v1[i-1];
if (y>0 && mask[i_mask] && mask[i_mask-width]) v2y = v2[i] - v2[i-width];
divv = -( v1x + v2y );
}
__global__ void g_updatePrimalVar(float *u, float *u_bar, float *u_diff, const float *p, const float *f, const float *scalar_op, float tau, float theta, size_t width, size_t height, size_t channels, const bool *mask) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;

if(x>=width || y>=height) return;
if(!mask[x+y*width]) return;

for(int c = 0; c < channels; c++) {
const size_t i = x + y * width + c * width * height;
const float u_old = u[i];

float divp;
d_calcDivergence( &p[0], &p[width*height*channels], divp, width, height, c, mask );

const float u_new = d_square_prox(u_old - tau * divp, scalar_op[i], f[i], tau);
u_bar[i] = u_new + theta * (u_new - u_old);
u[i] = u_new;
u_diff[i] = abs(u_new - u_old);
}
}