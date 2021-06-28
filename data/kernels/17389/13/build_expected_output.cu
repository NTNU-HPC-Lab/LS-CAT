#include "includes.h"
__global__ void build_expected_output(int *output, int n_rows, int k, const int *labels) {
int row = threadIdx.x + blockDim.x * blockIdx.x;
if (row >= n_rows) return;

int cur_label = labels[row];
for (int i = 0; i < k; i++) {
output[row * k + i] = cur_label;
}
}