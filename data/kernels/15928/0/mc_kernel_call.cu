#include "includes.h"



// wrapper pour une option d'achat
__global__ void mc_kernel_call(float * d_s, float T, float K, float S0, float sigma, float mu, float r, float dt, float * d_normals, unsigned N_STEPS, unsigned N_PATHS)
{
const unsigned tid = threadIdx.x; // id du thread dans le bloc
const unsigned bid = blockIdx.x; // id du bloc
const unsigned bsz = blockDim.x; // taille du bloc

int s_idx = tid + bid * bsz;
int n_idx = tid + bid * bsz;
float s_curr = S0;

if (s_idx < N_PATHS) {
int n = 0;
do {
s_curr = s_curr + mu*s_curr*dt + sigma*s_curr*d_normals[n_idx];
n_idx++;
n++;
} while (n < N_STEPS);
double payoff = (s_curr>K ? s_curr - K : 0.0);
__syncthreads(); // on attend que tous les threads aient fini avant de passer à la prochaine simulation
d_s[s_idx] = exp(-r*T) * payoff;
}
}