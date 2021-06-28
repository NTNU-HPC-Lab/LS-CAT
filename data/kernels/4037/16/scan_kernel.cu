#include "includes.h"
__global__ void scan_kernel(unsigned int* d_bins, int size) {
int mid = threadIdx.x + blockDim.x * blockIdx.x;
if (mid >= size) return;

for (int s = 1; s <= size; s *= 2) {
int spot = mid - s;

unsigned int val = 0;
if (spot >= 0) val = d_bins[spot];
__syncthreads();
if (spot >= 0) d_bins[mid] += val;
__syncthreads();
}
}