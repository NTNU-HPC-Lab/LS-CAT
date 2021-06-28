#include "includes.h"
__global__ void update_priority(int *new_priority, int n_selected, const int *new_idx, int n_ws, const int *idx, const int *priority) {
int tid = threadIdx.x + blockIdx.x * blockDim.x;
if (tid < n_selected) {
int my_new_idx = new_idx[tid];
// The working set size is limited (~1024 elements) so we just loop through it
for (int i = 0; i < n_ws; i++) {
if (idx[i] == my_new_idx) new_priority[tid] = priority[i] + 1;
}
}
}