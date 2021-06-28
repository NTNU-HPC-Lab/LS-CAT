#include "includes.h"
__global__ void prescan(float* d_in, int nGlobe, int step, int upSweep) {
int tid = blockDim.x * blockIdx.x + threadIdx.x;
int from = 2 * tid * (step + 1) + step;
int to = 2 * tid * (step + 1) + 2 * step + 1;
if (upSweep) {
d_in[to] += d_in[from];
} else {
int temp = d_in[to];
d_in[to] += d_in[from];
d_in[from] = temp;
}
}