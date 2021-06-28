#include "includes.h"


__global__ void diff_reduce(double *dev_w, double *feat, double *pos, int feat_dim, int pos_dim, int par0, int par1, int n_patch)
{
int i = blockIdx.y * blockDim.y + threadIdx.y;
int j = blockIdx.x * blockDim.x + threadIdx.x;

double feat_dist = 0.0; // running entry sum of d_ij
double pos_dist = 0.0;  // running entry sum of f_ij
int feat_offi = i * feat_dim; // offset of x_i
int feat_offj = j * feat_dim; // offset of x_j
int pos_offi = i * pos_dim;   // offset of p_i
int pos_offj = j * pos_dim;   // offset of p_j
double feat_i, feat_j, pos_i, pos_j;
// temporary local variables for entry sum calculation
int k;

if (i == j || i >= n_patch || j >= n_patch)
return;

/* thread (i, j) computes W_ij */

// get the k-th element of difference vector d_ij
// and add it to feat_dist
for (k = 0; k < feat_dim; k++) {
feat_i = feat[feat_offi + k];
feat_j = feat[feat_offj + k];
feat_dist += (feat_i - feat_j) * (feat_i - feat_j);
}

// get the k-th element of difference vector f_ij
// and add it to pos_dist
for (k = 0; k < pos_dim; k++) {
pos_i = pos[pos_offi + k];
pos_j = pos[pos_offj + k];
pos_dist += (pos_i - pos_j) * (pos_i - pos_j);
}

dev_w[i + j * n_patch]
= exp( -feat_dist / (feat_dim * par0 * par0))
* exp( -pos_dist / (pos_dim * par1 * par1));
}