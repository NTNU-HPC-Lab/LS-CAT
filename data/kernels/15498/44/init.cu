#include "includes.h"
__global__ void init(int n, float *x, float *y) {

int lane_id = threadIdx.x & 31;
size_t warp_id = (threadIdx.x + blockIdx.x * blockDim.x) >> 5;
size_t warps_per_grid = (blockDim.x * gridDim.x) >> 5;
size_t warp_total = ((sizeof(float)*n) + STRIDE_64K-1) / STRIDE_64K;


if(blockIdx.x==0 && threadIdx.x==0) {
//printf("\n TId[%d] ", threadIdx.x);
//printf(" WId[%u] ", warp_id);
//printf(" LId[%u] ", lane_id);
//printf(" WperG[%u] ", warps_per_grid);
//printf(" wTot[%u] ", warp_total);
//printf(" rep[%d] ", STRIDE_64K/sizeof(float)/32);
}
for(; warp_id < warp_total; warp_id += warps_per_grid) {
#pragma unroll
for(int rep = 0; rep < STRIDE_64K/sizeof(float)/32; rep++) {
size_t ind = warp_id * STRIDE_64K/sizeof(float) + rep * 32 + lane_id;
if (ind < n) {
x[ind] = 1.0f;
//if(blockIdx.x==0 && threadIdx.x==0) {
//	printf(" \nind[%d] ", ind);
//}
y[ind] = 2.0f;
}
}
}

}