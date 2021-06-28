#include "includes.h"
__global__ void cuda_sum_dist_compute(int dy, int dx, int nx, int ny, const int32_t* ones, uint32_t* sum_dist, int p)
{
int nx0      = blockIdx.x * blockDim.x + threadIdx.x;
int nxstride = blockDim.x * gridDim.x;
int dx0      = blockIdx.y * blockDim.y + threadIdx.y;
int dxstride = blockDim.y * gridDim.y;
int dy0      = blockIdx.z * blockDim.z + threadIdx.z;
int dystride = blockDim.z * gridDim.z;

for(int s = dy0; s < dy; s += dystride)
{
for(int d = dx0; d < dx; d += dxstride)
{
uint32_t*      _sum_dist = sum_dist + (s * nx * ny) + (d * nx);
const int32_t* _ones     = ones + (d * nx);
for(int n = nx0; n < nx; n += nxstride)
{
atomicAdd(&_sum_dist[n], (_ones[n] > 0) ? 1 : 0);
}
}
}
}