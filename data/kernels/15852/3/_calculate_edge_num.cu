#include "includes.h"
__global__ void _calculate_edge_num( long* edge_num, const long* edge_num_sum, const long* edge_idx_sort, const int b, const int n, const int orig_p_num, const long p_num ) {
int index = threadIdx.x + blockIdx.x * blockDim.x;
if (index >= b * n)
return;

const int c_b = index / n;
const int c_n = index % n;

long* c_edge_num = &edge_num[c_b * n * orig_p_num + c_n * orig_p_num];
const long c_edge_num_sum = edge_num_sum[c_b * n + c_n];
const long* c_edge_idx_sort = &edge_idx_sort[c_b * n * orig_p_num + c_n * orig_p_num];

if (c_edge_num_sum == p_num)
return;

if (c_edge_num_sum < p_num)
c_edge_num[c_edge_idx_sort[0]] += p_num - c_edge_num_sum;
else {
int id = 0;
long pass_num = c_edge_num_sum - p_num;
while (pass_num > 0) {
long edge_idx = c_edge_idx_sort[id];
if (c_edge_num[edge_idx] > pass_num) {
c_edge_num[edge_idx] -= pass_num;
pass_num = 0;
} else {
pass_num -= c_edge_num[edge_idx] - 1;
c_edge_num[edge_idx] = 1;
id += 1;
}
}
}
}