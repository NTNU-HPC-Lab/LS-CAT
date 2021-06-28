#include "includes.h"
__global__  void get_dists_kernel(const int * beg_pos, const int* adj_list, const int * weights, bool * mask, int* dists, int * update_dists, const int num_vtx) {

int tid = blockIdx.x*blockDim.x + threadIdx.x;
if (tid < num_vtx) {
if (mask[tid] == true) {
mask[tid] = false;
for (int edge = beg_pos[tid]; edge < beg_pos[tid + 1]; edge++) {
int other = adj_list[edge];
atomicMin(&update_dists[other],
dists[tid] + weights[edge]);
}
}
}
}