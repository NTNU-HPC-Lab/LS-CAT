#include "includes.h"
__global__ void reduce_ws(float *gdata, float *out){
__shared__ float sdata[32];
int tid = threadIdx.x;
int idx = threadIdx.x+blockDim.x*blockIdx.x;
float val = 0.0f;
unsigned mask = 0xFFFFFFFFU;
int lane = threadIdx.x % warpSize;
int warpID = threadIdx.x / warpSize;
while (idx < N) {  // grid stride loop to load
val += gdata[idx];
idx += gridDim.x*blockDim.x;
}

// 1st warp-shuffle reduction
for (int offset = warpSize/2; offset > 0; offset >>= 1)
val += __shfl_down_sync(mask, val, offset);
if (lane == 0) sdata[warpID] = val;
__syncthreads(); // put warp results in shared mem

// hereafter, just warp 0
if (warpID == 0){
// reload val from shared mem if warp existed
val = (tid < blockDim.x/warpSize)?sdata[lane]:0;

// final warp-shuffle reduction
for (int offset = warpSize/2; offset > 0; offset >>= 1)
val += __shfl_down_sync(mask, val, offset);

if  (tid == 0) atomicAdd(out, val);
}
}