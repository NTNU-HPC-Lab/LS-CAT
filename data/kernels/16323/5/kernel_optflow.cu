#include "includes.h"
__device__ size_t GIDX(size_t row, size_t col, int H, int W) {
return row * W + col;
}
__global__ void kernel_optflow(float* d_dx1, float* d_dy1, float* d_dx2, float* d_dy2, float* d_dt, float4* uv, float4* uv1, int H, int W) {

const size_t row = threadIdx.y + blockDim.y * blockIdx.y;
const size_t col = threadIdx.x + blockDim.x * blockIdx.x;
const size_t idx = GIDX(row, col, H, W);


if (row >= H - 2 || row <= 2 || col >= W - 2 || col <= 2) {
return;
}
__syncthreads();


float dx2 = 0.0f, dy2 = 0.0f;
float dxdy = 0.0f;
float dxdt = 0.0f, dydt = 0.0f;

for (int i = -2; i <= 2; i++) {
for (int j = -2; j <= 2; j++) {
dx2 += d_dx1[GIDX(row + i, col + j, H, W)] * d_dx1[GIDX(row + i, col + j, H, W)];
dy2 += d_dy1[GIDX(row + i, col + j, H, W)] * d_dy1[GIDX(row + i, col + j, H, W)];

dxdy += d_dx1[GIDX(row + i, col + j, H, W)] * d_dy1[GIDX(row + i, col + j, H, W)];

dxdt += d_dx1[GIDX(row + i, col + j, H, W)] * d_dt[GIDX(row + i, col + j, H, W)];
dydt += d_dy1[GIDX(row + i, col + j, H, W)] * d_dt[GIDX(row + i, col + j, H, W)];
}
}

__syncthreads();
float det = dx2 * dy2 - (dxdy * dxdy);
if (abs(det) <= 1.5e-8) { // 1.5e-5 is based on 1/(255*255)
uv[idx].x = 0.0f;
uv[idx].y = 0.0f;
uv1[idx] = uv[idx];
return;
}

__syncthreads();
float trace = dx2 + dy2;
float delta = sqrtf(trace * trace - 4.0f * det); // delta x2

if (isnan(delta) || trace - delta <= 0.0002) {
uv[idx].x = 0.0f;
uv[idx].y = 0.0f;
uv1[idx] = uv[idx];
return;
}

__syncthreads();
// Calculate flow components

uv[idx].x = (dy2 * -dxdt + dxdy * dydt)/det;
uv[idx].y = (dxdy * dxdt - dx2 * dydt)/ det;
uv1[idx] = uv[idx];

}