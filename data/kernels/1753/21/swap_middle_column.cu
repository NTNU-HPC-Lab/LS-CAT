#include "includes.h"
__global__ void swap_middle_column(float* data, const int num_threads, const int nx, const int ny, const int xodd, const int yodd, const int offset) {
const uint x=threadIdx.x;
const uint y=blockIdx.x;

const uint r = x+y*num_threads+offset;
int c = nx/2;
int idx1 = r*nx + c;
int idx2 = (r+ny/2+yodd)*nx + c;
float tmp = data[idx1];
data[idx1] = data[idx2];
data[idx2] = tmp;
}