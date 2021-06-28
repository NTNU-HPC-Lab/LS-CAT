#include "includes.h"
__global__ void make_pillar_feature_kernel( float* dev_pillar_point_feature_in_coors, float* dev_pillar_point_feature, float* dev_pillar_coors, int* dev_x_coors, int* dev_y_coors, float* dev_num_points_per_pillar, const int max_points, const int num_point_feature, const int grid_x_size) {
int ith_pillar = blockIdx.x;
int num_points_at_this_pillar = dev_num_points_per_pillar[ith_pillar];
int ith_point = threadIdx.x;
if (ith_point >= num_points_at_this_pillar) {
return;
}
int x_ind = dev_x_coors[ith_pillar];
int y_ind = dev_y_coors[ith_pillar];
int pillar_ind = ith_pillar * max_points * num_point_feature +
ith_point * num_point_feature;
int coors_ind = y_ind * grid_x_size * max_points * num_point_feature +
x_ind * max_points * num_point_feature +
ith_point * num_point_feature;
for (int i = 0; i < num_point_feature; ++i) {
dev_pillar_point_feature[pillar_ind + i] =
dev_pillar_point_feature_in_coors[coors_ind + i];
}

float coor_x = static_cast<float>(x_ind);
float coor_y = static_cast<float>(y_ind);
dev_pillar_coors[ith_pillar * 4 + 0] = 0;  // batch idx
dev_pillar_coors[ith_pillar * 4 + 1] = 0;  // z
dev_pillar_coors[ith_pillar * 4 + 2] = coor_y;
dev_pillar_coors[ith_pillar * 4 + 3] = coor_x;
}