#include "includes.h"
__global__ void sum_V ( float *V, float *V_s, int num_atom, int num_atom2, int *Ele, float *vdW) {

for (int ii = threadIdx.x; ii < num_atom2; ii += blockDim.x) {
if (ii < num_atom) {
int atomi = Ele[ii];
if (atomi > 5) atomi = 0;
V_s[ii] = V[ii] * 4.0 * PI * vdW[atomi] * vdW[atomi];
} else {
V_s[ii] = 0.0;
}
}
for (int stride = num_atom2 / 2; stride > 0; stride >>= 1) {
__syncthreads();
for(int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x) {
V_s[iAccum] += V_s[stride + iAccum];
}
}
__syncthreads();
if (threadIdx.x == 0) printf("Convex contact area = %.3f A^2.\n", V_s[0]);
}