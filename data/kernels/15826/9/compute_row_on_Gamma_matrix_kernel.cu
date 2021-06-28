#include "includes.h"
__global__ void compute_row_on_Gamma_matrix_kernel(int row_index, int vertex_index, int* indices, double* exp_V, double* N_ptr, int LD_N, double* G_ptr, int LD_G, double* row_ptr, int incr) {
// int l = threadIdx.x;
int l = blockIdx.x;

int i_index, j_index;
double delta;

i_index = indices[row_index];
j_index = indices[l];

if (j_index < vertex_index) {
delta = i_index == j_index ? 1 : 0;
row_ptr[l * incr] = (N_ptr[i_index + LD_N * j_index] * exp_V[l] - delta) / (exp_V[l] - 1.);
}
else
row_ptr[l * incr] = G_ptr[i_index + LD_G * (j_index - vertex_index)];
}