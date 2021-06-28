#include "includes.h"
__global__ void compute_col_on_Gamma_matrix_kernel(int col_index, int vertex_index, int* indices, double* exp_V, double* N_ptr, int LD_N, double* G_ptr, int LD_G, double* col_ptr, int incr) {
// int l = threadIdx.x;
int l = blockIdx.x;

int i_index, j_index;
double delta, exp_Vj;

i_index = indices[l];
j_index = indices[col_index];

exp_Vj = exp_V[col_index];

if (j_index < vertex_index) {
delta = i_index == j_index ? 1 : 0;
col_ptr[l * incr] = (N_ptr[i_index + LD_N * j_index] * exp_Vj - delta) / (exp_Vj - 1.);
}
else
col_ptr[l * incr] = G_ptr[i_index + LD_G * (j_index - vertex_index)];
}