#include "includes.h"
__global__ void calc_entropy_atomic(float *float_image_in, float *entropy_out, int blk_size) {
//calculate entropy of a block through a single thread
__shared__ float sum;
if (threadIdx.x == 0 && threadIdx.y == 0) {
sum = 0.0;
}
__syncthreads();
int blocksize = blk_size*blk_size;
//vertical offset to get to beginning of own block
int v_offset_to_blkrow = gridDim.x*blockDim.x*blockDim.y*blockIdx.y;
int v_offset_to_pixrow = blockDim.x*gridDim.x*threadIdx.y;
int h_offset = blockDim.x*blockIdx.x + threadIdx.x;
int idx = v_offset_to_blkrow + v_offset_to_pixrow + h_offset; //idx of top left corner of the block
int out_idx = blockIdx.y*gridDim.x + blockIdx.x;
//normalize image
float_image_in[idx] = float_image_in[idx] * float_image_in[idx] / (blocksize);
atomicAdd(&sum, float_image_in[idx]);
__syncthreads();
__shared__ float entropy;
if (threadIdx.x == 0 && threadIdx.y == 0) {
entropy = 0.0;
}
__syncthreads();
float_image_in[idx] = float_image_in[idx] / sum;
//shannon entropy
atomicAdd(&entropy, -float_image_in[idx] * log2(float_image_in[idx]));
__syncthreads();
//printf("%f\n", sum2);
if (threadIdx.x == 0 && threadIdx.y == 0) {
entropy_out[out_idx] = entropy;
}
}