#include "includes.h"
__global__ void compute_G_cols_kernel(int N_i, int N_r, int N_c, int* p_ptr, double* exp_V_ptr, double* N_ptr, int N_ld, double* G_ptr, int G_ld, double* G_cols_ptr, int G_cols_ld) {
int I = threadIdx.x + blockIdx.x * BLOCK_SIZE_x;  // blockDim.x;

int l_MIN = BLOCK_SIZE_y * (blockIdx.y + 0);
int l_MAX = BLOCK_SIZE_y * (blockIdx.y + 1);

l_MIN = max(l_MIN, 0);
l_MAX = min(l_MAX, N_i);

if (I < N_r) {
// for(int l=0; l<N_i; ++l)
for (int l = l_MIN; l < l_MAX; ++l) {
if (p_ptr[l] >= N_c) {
G_cols_ptr[I + l * G_cols_ld] = G_ptr[I + (p_ptr[l] - N_c) * G_ld];
}
else {
double alpha = exp_V_ptr[l] / (exp_V_ptr[l] - 1.);

G_cols_ptr[I + l * G_cols_ld] = alpha * N_ptr[I + p_ptr[l] * N_ld];
}
}

// for(int l=0; l<N_i; ++l)
for (int l = l_MIN; l < l_MAX; ++l)
if (p_ptr[l] < N_c and I == p_ptr[l])
G_cols_ptr[I + l * G_cols_ld] -= 1. / (exp_V_ptr[l] - 1.);
}
}