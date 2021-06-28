#include "includes.h"
__global__ void make_extra_network_input_kernel(float* dev_x_coors_for_sub, float* dev_y_coors_for_sub, float* dev_num_points_per_pillar, float* dev_x_coors_for_sub_shaped, float* dev_y_coors_for_sub_shaped, float* dev_pillar_feature_mask, const int MAX_NUM_POINTS_PER_PILLAR)
{
int ith_pillar = blockIdx.x;
int ith_point = threadIdx.x;
float x = dev_x_coors_for_sub[ith_pillar];
float y = dev_y_coors_for_sub[ith_pillar];
int num_points_for_a_pillar = dev_num_points_per_pillar[ith_pillar];
int ind = ith_pillar*MAX_NUM_POINTS_PER_PILLAR + ith_point;
dev_x_coors_for_sub_shaped[ind] = x;
dev_y_coors_for_sub_shaped[ind] = y;

if(ith_point < num_points_for_a_pillar)
{
dev_pillar_feature_mask[ind] = 1.0;
}
else
{
dev_pillar_feature_mask[ind] = 0.0;
}
}