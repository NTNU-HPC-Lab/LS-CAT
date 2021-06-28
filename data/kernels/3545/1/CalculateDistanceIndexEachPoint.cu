#include "includes.h"
// Device code for ICP computation
// Currently working only on performing rotation and translation using cuda


#ifndef _ICP_KERNEL_H_
#define _ICP_KERNEL_H_



#define TILE_WIDTH 256




















#endif // #ifndef _ICP_KERNEL_H_
__global__ void CalculateDistanceIndexEachPoint(double point_x, double point_y, double point_z, double * data_x_d, double * data_y_d, double * data_z_d, int * bin_index_d, double * distance_d, int size_data)
{
int index = blockDim.x*blockIdx.x + threadIdx.x;
if(index < size_data)
{
distance_d[index] = sqrt(pow(data_x_d[index] - point_x,2) + pow(data_y_d[index] - point_y,2) + pow(data_z_d[index] - point_z,2));
bin_index_d[index] = index;
}

}