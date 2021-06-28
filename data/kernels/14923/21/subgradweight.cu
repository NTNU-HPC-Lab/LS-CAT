#include "includes.h"
__global__ void subgradweight(float *input, float *gradOutput, float *gradWeight, float *gradBias, int input_n, int input_h, int input_w, int kH, int kW, int dH, int dW, float scale)
{
// iterators
int xx, yy;

// output size
int output_w = (input_w - kW) / dW + 1;
int output_h = (input_h - kH) / dH + 1;

// compute offsets based on thread/block ID
int o = blockIdx.x;
int i = o;
int k = blockIdx.x % input_n;

int xx_start = threadIdx.x;
int xx_end = output_w;
int xx_step = blockDim.x;

int yy_start = threadIdx.y;
int yy_end = output_h;
int yy_step = blockDim.y;

// select input/output plane
gradOutput = gradOutput + o*output_w*output_h;
input = input + i*input_w*input_h;

// thread ID
int tid = blockDim.x*threadIdx.y + threadIdx.x;

// create array to hold partial sums
__shared__ float sums[CUDA_MAX_THREADS];
sums[tid] = 0;

// compute partial sums
for(yy = yy_start; yy < yy_end; yy+=yy_step) {
for(xx = xx_start; xx < xx_end; xx+=xx_step) {
float *ptr_input = input + yy*dH*input_w + xx*dW;
float *ptr_gradOutput = gradOutput + yy*output_w + xx;
float z = *ptr_gradOutput;
long kx, ky;
for(ky = 0; ky < kH; ky++) {
for(kx = 0; kx < kW; kx++) {
sums[tid] += z * ptr_input[kx];
}
ptr_input += input_w;
}
}
}
__syncthreads();

// reduce: accumulate all partial sums to produce final gradWeight
if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
for(int i = 0; i < blockDim.x*blockDim.y; i++) gradWeight[k] += scale*sums[i];
}
__syncthreads();

// compute gradBias
sums[tid] = 0;
for (int i=tid; i<output_w*output_h; i+=(blockDim.x*blockDim.y)) {
sums[tid] += gradOutput[i];
}
__syncthreads();

// reduce gradBias
if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
for (int i=0; i<(blockDim.x*blockDim.y); i++)
gradBias[k] += scale*sums[i];
}
}