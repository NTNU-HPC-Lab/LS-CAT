#include "includes.h"
__global__ void create_FF_full_FoXS_surf_grad ( float *FF_table, float *V, float c2, int *Ele, float *FF_full, float *surf_grad, int num_q, int num_ele, int num_atom, int num_atom2) {

__shared__ float FF_pt[7];
float hydration;
for (int ii = blockIdx.x; ii < num_q+1; ii += gridDim.x) {

// Get form factor for this block (or q vector)
if (ii < num_q) {
for (int jj = threadIdx.x; jj < num_ele + 1; jj += blockDim.x) {
FF_pt[jj] = FF_table[ii*(num_ele+1)+jj];
}
}
__syncthreads();

// In FoXS since c2 remains the same for all elements it is reduced to one value.
hydration = c2 * FF_pt[num_ele];
//if (ii == num_q && threadIdx.x == 0) {printf("Hydration is: %6.3f\n", hydration);}
__syncthreads();
// Calculate atomic form factor for this q
// However to keep compatible to HyPred method we leave atom type def unchanged.
if (ii == num_q) {
// calculate surf_grad
for (int jj = threadIdx.x; jj < num_atom; jj += blockDim.x) {
//int atomt = Ele[jj];
//printf("B surf grads = %6.3f, %6.3f, %6.3f. \n",
//       surf_grad[3*jj], surf_grad[3*jj+1], surf_grad[3*jj+2]);
/*surf_grad[3*jj]   *= hydration;
surf_grad[3*jj+1] *= hydration;
surf_grad[3*jj+2] *= hydration;*/
surf_grad[3*jj]   *= c2;
surf_grad[3*jj+1] *= c2;
surf_grad[3*jj+2] *= c2;
//printf("A surf grads = %6.3f, %6.3f, %6.3f. \n",
//       surf_grad[3*jj], surf_grad[3*jj+1], surf_grad[3*jj+2]);
}
} else {
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
if (threadIdx.x == 0) FF_full[ii * num_atom2 + num_atom + 1] = FF_pt[num_ele];
}
}