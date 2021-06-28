#include "includes.h"
__global__ void subsample_ind_and_labels_GPU(int *d_ind_sub, const int *d_ind, unsigned int *d_label_sub, const unsigned int *d_label, int n_out, float inv_sub_factor) {

unsigned int ind_out = blockIdx.x * blockDim.x + threadIdx.x;

if (ind_out < n_out) {

int ind_in = (int)floorf((float)(ind_out) * inv_sub_factor);
d_ind_sub[ind_out] = d_ind[ind_in];
d_label_sub[ind_out] = d_label[ind_in];
}
}