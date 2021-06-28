#include "includes.h"
__global__ void _calculate_wnp( const long* edge_num, const long* edge_start_idx, float* weight, long* ind, const int b, const int n, const int orig_p_num, const int p_num ) {
int index = threadIdx.x + blockIdx.x * blockDim.x;
if (index >= b * n * orig_p_num)
return;

const int c_b = index / (n * orig_p_num);
const int c_n = (index - c_b * n * orig_p_num) / orig_p_num;
const int c_edge_idx = index % orig_p_num;

const long c_edge_num = edge_num[index];
const int c_start_idx = int(edge_start_idx[index]);
float* c_weight = &weight[c_b * n * p_num + c_n * p_num + c_start_idx];
long* c_ind = &ind[c_b * n * p_num * 2 + c_n * p_num * 2 + c_start_idx * 2];

for (long i = 0; i < c_edge_num; i++) {
c_weight[i] = float(i) / float(c_edge_num);
c_ind[i * 2] = long(c_edge_idx);
c_ind[i * 2 + 1] = long((c_edge_idx + 1) % orig_p_num);
}
}