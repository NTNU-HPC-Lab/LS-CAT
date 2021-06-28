#include "includes.h"
__global__ void swap_bot_left_top_right(float* data, const int num_threads, const int nx, const int ny, const int xodd, const int yodd, const int offset) {
const uint x=threadIdx.x;
const uint y=blockIdx.x;

const uint gpu_idx = x+y*num_threads+offset;
const uint c = gpu_idx % (nx/2);
const uint r = gpu_idx / (nx/2);

const uint idx1 = r*nx + c;
const uint idx2 = (r+ny/2+yodd)*nx + c + nx/2+xodd;
float tmp = data[idx1];
data[idx1] = data[idx2];
data[idx2] = tmp;
}