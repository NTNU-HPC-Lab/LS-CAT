#include "includes.h"
__global__ void read_G_matrix_kernel(int S, int vertex_index, int* i_index, int* j_index, bool* is_Bennett, double* exp_Vj, double* N_ptr, int LD_N, double* G_ptr, int LD_G, double* result_ptr, int incr) {
int l = threadIdx.x;

double result, delta;

if (j_index[l] < vertex_index) {
delta = i_index[l] == j_index[l] ? 1. : 0.;
result = (N_ptr[i_index[l] + LD_N * j_index[l]] * exp_Vj[l] - delta) / (exp_Vj[l] - 1.);
}
else
result = G_ptr[i_index[l] + LD_G * (j_index[l] - vertex_index)];

result_ptr[l * incr] = is_Bennett[l] ? 0. : result;
}