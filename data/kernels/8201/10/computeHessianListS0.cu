#include "includes.h"
__global__ void computeHessianListS0(float *trans_x, float *trans_y, float *trans_z, int *valid_points, int *starting_voxel_id, int *voxel_id, int valid_points_num, double *centroid_x, double *centroid_y, double *centroid_z, double *icov00, double *icov01, double *icov02, double *icov10, double *icov11, double *icov12, double *icov20, double *icov21, double *icov22, double *point_gradients0, double *point_gradients1, double *point_gradients2, double *tmp_hessian, int valid_voxel_num)
{
int id = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
int col = blockIdx.y;

if (col < 6) {
double *tmp_pg0 = point_gradients0 + col * valid_points_num;
double *tmp_pg1 = point_gradients1 + 6 * valid_points_num;
double *tmp_pg2 = point_gradients2 + 6 * valid_points_num;
double *tmp_h = tmp_hessian + col * valid_voxel_num;

for (int i = id; i < valid_points_num && col < 6; i += stride) {
int pid = valid_points[i];
double d_x = static_cast<double>(trans_x[pid]);
double d_y = static_cast<double>(trans_y[pid]);
double d_z = static_cast<double>(trans_z[pid]);

double pg0 = tmp_pg0[i];
double pg1 = tmp_pg1[i];
double pg2 = tmp_pg2[i];

for ( int j = starting_voxel_id[i]; j < starting_voxel_id[i + 1]; j++) {
int vid = voxel_id[j];

tmp_h[j] = (d_x - centroid_x[vid]) * (icov00[vid] * pg0 + icov01[vid] * pg1 + icov02[vid] * pg2)
+ (d_y - centroid_y[vid]) * (icov10[vid] * pg0 + icov11[vid] * pg1 + icov12[vid] * pg2)
+ (d_z - centroid_z[vid]) * (icov20[vid] * pg0 + icov21[vid] * pg1 + icov22[vid] * pg2);
}
}
}
}