#include "includes.h"
__global__ void computeHessianListS2(float *trans_x, float *trans_y, float *trans_z, int *valid_points, int *starting_voxel_id, int *voxel_id, int valid_points_num, double *centroid_x, double *centroid_y, double *centroid_z, double gauss_d1, double *e_x_cov_x, double *icov00, double *icov01, double *icov02, double *icov10, double *icov11, double *icov12, double *icov20, double *icov21, double *icov22, double *point_hessians, double *hessians, int valid_voxel_num)
{
int id = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
int row = blockIdx.y;
int col = blockIdx.z;

if (row < 6 && col < 6) {
double *h = hessians + (row * 6 + col) * valid_points_num;
double *tmp_ph0 = point_hessians + ((3 * row) * 6 + col) * valid_points_num;
double *tmp_ph1 = tmp_ph0 + 6 * valid_points_num;
double *tmp_ph2 = tmp_ph1 + 6 * valid_points_num;

for (int i = id; i < valid_points_num; i += stride) {
int pid = valid_points[i];
double d_x = static_cast<double>(trans_x[pid]);
double d_y = static_cast<double>(trans_y[pid]);
double d_z = static_cast<double>(trans_z[pid]);
double ph0 = tmp_ph0[i];
double ph1 = tmp_ph1[i];
double ph2 = tmp_ph2[i];

double final_hessian = h[i];

for ( int j = starting_voxel_id[i]; j < starting_voxel_id[i + 1]; j++) {
//Transformed coordinates
int vid = voxel_id[j];
double tmp_ex = e_x_cov_x[j];

if (!(tmp_ex > 1 || tmp_ex < 0 || tmp_ex != tmp_ex)) {
tmp_ex *= gauss_d1;

final_hessian += (d_x - centroid_x[vid]) * (icov00[vid] * ph0 + icov01[vid] * ph1 + icov02[vid] * ph2) * tmp_ex;
final_hessian += (d_y - centroid_y[vid]) * (icov10[vid] * ph0 + icov11[vid] * ph1 + icov12[vid] * ph2) * tmp_ex;
final_hessian += (d_z - centroid_z[vid]) * (icov20[vid] * ph0 + icov21[vid] * ph1 + icov22[vid] * ph2) * tmp_ex;

}
}

h[i] = final_hessian;
}
}
}