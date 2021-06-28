#include "includes.h"
__global__  void update_dists_kernel(const int * beg_pos, const int * adj_list, const int* weights, bool * mask, int* dists, int* update_dists, const int num_vtx) {

int tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid < num_vtx) {
if (dists[tid] > update_dists[tid]) {
dists[tid] = update_dists[tid];
mask[tid] = true;
}
update_dists[tid] = dists[tid];
}
}