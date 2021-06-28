#include "includes.h"
__global__ void scatter_kernel( int *x_coors, int *y_coors, float *pfe_output, float *scattered_feature, const int MAX_NUM_PILLARS_, const int GRID_X_SIZE, const int GRID_Y_SIZE)
{
int i_pillar = blockIdx.x;
int i_feature = threadIdx.x;
int x_ind = x_coors[i_pillar];
int y_ind = y_coors[i_pillar];
float feature = pfe_output[i_feature*MAX_NUM_PILLARS_ + i_pillar];
scattered_feature[i_feature*GRID_Y_SIZE*GRID_X_SIZE + y_ind * GRID_X_SIZE + x_ind] = feature;
}