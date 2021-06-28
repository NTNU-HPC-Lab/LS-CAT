#include "includes.h"
__global__ void make_pillar_histo_kernel( const float* dev_points, float* dev_pillar_x_in_coors, float* dev_pillar_y_in_coors, float* dev_pillar_z_in_coors, float* dev_pillar_i_in_coors, int* pillar_count_histo, const int num_points, const int max_points_per_pillar, const int GRID_X_SIZE, const int GRID_Y_SIZE, const int GRID_Z_SIZE, const float MIN_X_RANGE, const float MIN_Y_RANGE, const float MIN_Z_RANGE, const float PILLAR_X_SIZE, const float PILLAR_Y_SIZE, const float PILLAR_Z_SIZE, const int NUM_BOX_CORNERS )
{
int th_i = threadIdx.x + blockIdx.x * blockDim.x;
if(th_i >= num_points)
{
return;
}
int y_coor = floor((dev_points[th_i*NUM_BOX_CORNERS + 1] - MIN_Y_RANGE)/PILLAR_Y_SIZE);
int x_coor = floor((dev_points[th_i*NUM_BOX_CORNERS + 0] - MIN_X_RANGE)/PILLAR_X_SIZE);
int z_coor = floor((dev_points[th_i*NUM_BOX_CORNERS + 2] - MIN_Z_RANGE)/PILLAR_Z_SIZE);

if(x_coor >= 0 && x_coor < GRID_X_SIZE &&
y_coor >= 0 && y_coor < GRID_Y_SIZE &&
z_coor >= 0 && z_coor < GRID_Z_SIZE)
{
int count = atomicAdd(&pillar_count_histo[y_coor*GRID_X_SIZE + x_coor], 1);
if(count < max_points_per_pillar)
{
int ind = y_coor*GRID_X_SIZE*max_points_per_pillar + x_coor*max_points_per_pillar + count;
dev_pillar_x_in_coors[ind] = dev_points[th_i*NUM_BOX_CORNERS + 0];
dev_pillar_y_in_coors[ind] = dev_points[th_i*NUM_BOX_CORNERS + 1];
dev_pillar_z_in_coors[ind] = dev_points[th_i*NUM_BOX_CORNERS + 2];
dev_pillar_i_in_coors[ind] = dev_points[th_i*NUM_BOX_CORNERS + 3];
}
}
}