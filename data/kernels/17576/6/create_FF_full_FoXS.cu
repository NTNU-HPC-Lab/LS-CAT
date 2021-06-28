#include "includes.h"
__global__ void create_FF_full_FoXS ( float *FF_table, float *V, float c2, int *Ele, float *FF_full, int num_q, int num_ele, int num_atom, int num_atom2) {

__shared__ float FF_pt[7];
float hydration;
for (int ii = blockIdx.x; ii < num_q; ii += gridDim.x) {

// Get form factor for this block (or q vector)
if (ii < num_q) {
for (int jj = threadIdx.x; jj < num_ele + 1; jj += blockDim.x) {
FF_pt[jj] = FF_table[ii*(num_ele+1)+jj];
}
}
__syncthreads();

// In FoXS since c2 remains the same for all elements it is reduced to one value.
hydration = c2 * FF_pt[num_ele];

// Calculate atomic form factor for this q
// However to keep compatible to HyPred method we leave atom type def unchanged.
for (int jj = threadIdx.x; jj < num_atom; jj += blockDim.x) {
int atomt = Ele[jj];
if (atomt > 5) {  // Which means this is a hydrogen
FF_full[ii*num_atom2 + jj] = FF_pt[0];
FF_full[ii*num_atom2 + jj] += hydration * V[jj];
} else {          // Heavy atoms - do the same as before
FF_full[ii*num_atom2 + jj] = FF_pt[atomt];
FF_full[ii*num_atom2 + jj] += hydration * V[jj];
}
}
}
}