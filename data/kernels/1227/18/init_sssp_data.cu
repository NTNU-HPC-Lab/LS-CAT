#include "includes.h"
__global__ void init_sssp_data(bool * d_mask, int* d_dists, int* d_update_dists, const int source, const int num_vtx) {

int tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid < num_vtx) {
if (source == tid) {
d_mask[tid] = true;
d_dists[tid] = 0;
d_update_dists[tid] = 0;
}
else {
d_mask[tid] = false;
d_dists[tid] = INT_MAX;
d_update_dists[tid] = INT_MAX;
}
}
}