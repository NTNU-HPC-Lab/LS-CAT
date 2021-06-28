#include "includes.h"
__global__ void compute_Gamma_kernel(double* Gamma, int Gamma_n, int Gamma_ld, double* N, int N_r, int N_c, int N_ld, double* G, int G_r, int G_c, int G_ld, int* random_vertex_vector, double* exp_V, double* exp_delta_V) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;

int vertex_index = N_c - G_c;

if (i < Gamma_n and j < Gamma_n) {
int configuration_e_spin_index_i = random_vertex_vector[i];
int configuration_e_spin_index_j = random_vertex_vector[j];

if (configuration_e_spin_index_j < vertex_index) {
double delta = 0;

if (configuration_e_spin_index_i == configuration_e_spin_index_j)
delta = 1.;

double N_ij = N[configuration_e_spin_index_i + configuration_e_spin_index_j * N_ld];

Gamma[i + j * Gamma_ld] = (N_ij * exp_V[j] - delta) / (exp_V[j] - 1.);
}
else
Gamma[i + j * Gamma_ld] =
G[configuration_e_spin_index_i + (configuration_e_spin_index_j - vertex_index) * G_ld];
}

if (i < Gamma_n and j < Gamma_n and i == j) {
double gamma_k = exp_delta_V[j];
Gamma[i + j * Gamma_ld] -= (gamma_k) / (gamma_k - 1.);
}
}