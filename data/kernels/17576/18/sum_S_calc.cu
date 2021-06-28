#include "includes.h"
__global__ void sum_S_calc ( float *S_calcc, float *f_ptxc, float *f_ptyc, float *f_ptzc, float *S_calc, float *Aq, float *q_S_ref_dS, int num_q, int num_atom, int num_atom2, float alpha, float k_chi, float *sigma2) {

for (int ii = blockIdx.x; ii < num_q; ii += gridDim.x) {
// Tree-like summation of S_calcc to get S_calc
for (int stride = num_atom2 / 2; stride > 0; stride >>= 1) {
__syncthreads();
for(int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
S_calcc[ii * num_atom2 + iAccum] += S_calcc[ii * num_atom2 + stride + iAccum];
}
}
__syncthreads();

S_calc[ii] = S_calcc[ii * num_atom2];
__syncthreads();
if (threadIdx.x == 0) {
Aq[ii] = S_calc[ii] - q_S_ref_dS[ii+num_q];
Aq[ii] *= -alpha;
Aq[ii] += q_S_ref_dS[ii + 2*num_q];
Aq[ii] *= k_chi / sigma2[ii];
Aq[ii] += Aq[ii];
}
__syncthreads();
for (int jj = threadIdx.x; jj < num_atom; jj += blockDim.x) {
f_ptxc[ii * num_atom2 + jj] *= Aq[ii] * alpha;
f_ptyc[ii * num_atom2 + jj] *= Aq[ii] * alpha;
f_ptzc[ii * num_atom2 + jj] *= Aq[ii] * alpha;
}
}
}