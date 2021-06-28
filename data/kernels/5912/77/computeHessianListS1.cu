#include "includes.h"
__global__ void computeHessianListS1(float *trans_x, float *trans_y, float *trans_z, int *valid_points, int *starting_voxel_id, int *voxel_id, int valid_points_num, double *centroid_x, double *centroid_y, double *centroid_z, double gauss_d1, double gauss_d2, double *hessians, double *e_x_cov_x, double *tmp_hessian, double *cov_dxd_pi, double *point_gradients, int valid_voxel_num)
{
int id = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
int row = blockIdx.y;
int col = blockIdx.z;

if (row < 6 && col < 6) {
double *cov_dxd_pi_mat0 = cov_dxd_pi + row * valid_voxel_num;
double *cov_dxd_pi_mat1 = cov_dxd_pi_mat0 + 6 * valid_voxel_num;
double *cov_dxd_pi_mat2 = cov_dxd_pi_mat1 + 6 * valid_voxel_num;
double *tmp_h = tmp_hessian + col * valid_voxel_num;
double *h = hessians + (row * 6 + col) * valid_points_num;
double *tmp_pg0 = point_gradients + col * valid_points_num;
double *tmp_pg1 = tmp_pg0 + 6 * valid_points_num;
double *tmp_pg2 = tmp_pg1 + 6 * valid_points_num;

for (int i = id; i < valid_points_num; i += stride) {
int pid = valid_points[i];
double d_x = static_cast<double>(trans_x[pid]);
double d_y = static_cast<double>(trans_y[pid]);
double d_z = static_cast<double>(trans_z[pid]);

double pg0 = tmp_pg0[i];
double pg1 = tmp_pg1[i];
double pg2 = tmp_pg2[i];

double final_hessian = 0.0;

for ( int j = starting_voxel_id[i]; j < starting_voxel_id[i + 1]; j++) {
//Transformed coordinates
int vid = voxel_id[j];

double tmp_ex = e_x_cov_x[j];

if (!(tmp_ex > 1 || tmp_ex < 0 || tmp_ex != tmp_ex)) {
double cov_dxd0 = cov_dxd_pi_mat0[j];
double cov_dxd1 = cov_dxd_pi_mat1[j];
double cov_dxd2 = cov_dxd_pi_mat2[j];

tmp_ex *= gauss_d1;

final_hessian += -gauss_d2 * ((d_x - centroid_x[vid]) * cov_dxd0 + (d_y - centroid_y[vid]) * cov_dxd1 + (d_z - centroid_z[vid]) * cov_dxd2) * tmp_h[j] * tmp_ex;
final_hessian += (pg0 * cov_dxd0 + pg1 * cov_dxd1 + pg2 * cov_dxd2) * tmp_ex;
}
}

h[i] = final_hessian;
}
}
}