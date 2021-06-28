#include "includes.h"
__global__ void force_calc_EMA ( float *Force, double *Force_old, int num_atom, int num_q, float *f_ptxc, float *f_ptyc, float *f_ptzc, int num_atom2, int num_q2, int *Ele, double EMA_norm, float force_ramp) {
// Do column tree sum of f_ptxc for f_ptx for every atom, then assign threadIdx.x == 0 (3 * num_atoms) to Force. Force is num_atom * 3.
if (blockIdx.x >= num_atom) return;
for (int ii = blockIdx.x; ii < num_atom; ii += gridDim.x) {
for (int stride = num_q2 / 2; stride > 0; stride >>= 1) {
__syncthreads();
for(int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
f_ptxc[ii + iAccum * num_atom2] += f_ptxc[ii + iAccum * num_atom2 + stride * num_atom2];
f_ptyc[ii + iAccum * num_atom2] += f_ptyc[ii + iAccum * num_atom2 + stride * num_atom2];
f_ptzc[ii + iAccum * num_atom2] += f_ptzc[ii + iAccum * num_atom2 + stride * num_atom2];
}
}
__syncthreads();
if (threadIdx.x == 0) {
if (Ele[ii]) {
Force_old[ii*3    ] *= (EMA_norm - 1.0);
Force_old[ii*3    ] -= (double)f_ptxc[ii];
Force_old[ii*3    ] /= EMA_norm;
Force_old[ii*3 + 1] *= (EMA_norm - 1.0);
Force_old[ii*3 + 1] -= (double)f_ptyc[ii];
Force_old[ii*3 + 1] /= EMA_norm;
Force_old[ii*3 + 2] *= (EMA_norm - 1.0);
Force_old[ii*3 + 2] -= (double)f_ptzc[ii];
Force_old[ii*3 + 2] /= EMA_norm;
Force[ii*3    ] = (float)Force_old[ii*3    ] * force_ramp;
Force[ii*3 + 1] = (float)Force_old[ii*3 + 1] * force_ramp;
Force[ii*3 + 2] = (float)Force_old[ii*3 + 2] * force_ramp;
}
}
__syncthreads();
}
}