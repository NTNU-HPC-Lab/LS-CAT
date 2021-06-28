#include "includes.h"
__global__ void set_unavailable(bool *available, int n_rows, const int *idx, int n_selected) {
int tid = threadIdx.x + blockIdx.x * blockDim.x;
if (tid < n_selected) {
available[idx[tid]] = false;
}
}