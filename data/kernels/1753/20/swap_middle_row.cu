#include "includes.h"
__global__ void swap_middle_row(float* data, const int num_threads, const int nx, const int ny, const int xodd, const int yodd, const int offset) {
const uint x=threadIdx.x;
const uint y=blockIdx.x;

const uint c = x+y*num_threads+offset;
int r = ny/2;
int idx1 = r*nx + c;
int idx2 = r*nx + c + nx/2+ xodd;
float tmp = data[idx1];
data[idx1] = data[idx2];
data[idx2] = tmp;
}