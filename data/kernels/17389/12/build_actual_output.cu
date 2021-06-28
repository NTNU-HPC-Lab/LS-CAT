#include "includes.h"
__global__ void build_actual_output(int *output, int n_rows, int k, const int *idx_labels, const int64_t *indices) {
int element = threadIdx.x + blockDim.x * blockIdx.x;
if (element >= n_rows * k) return;

int ind = (int)indices[element];
output[element] = idx_labels[ind];
}