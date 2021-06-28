#include "includes.h"
__global__ void computeScoreList(int *starting_voxel_id, int *voxel_id, int valid_points_num, double *e_x_cov_x, double gauss_d1, double *score)
{
int id = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;

for (int i = id; i < valid_points_num; i += stride) {

double score_inc = 0;

for (int vid = starting_voxel_id[i]; vid < starting_voxel_id[i + 1]; vid++) {
double tmp_ex = e_x_cov_x[vid];

score_inc += (tmp_ex > 1 || tmp_ex < 0 || tmp_ex != tmp_ex) ? 0 : -gauss_d1 * tmp_ex;
}

score[i] = score_inc;
}
}