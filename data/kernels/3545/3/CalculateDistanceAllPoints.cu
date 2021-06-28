#include "includes.h"
// Device code for ICP computation
// Currently working only on performing rotation and translation using cuda


#ifndef _ICP_KERNEL_H_
#define _ICP_KERNEL_H_



#define TILE_WIDTH 256




















#endif // #ifndef _ICP_KERNEL_H_
__global__ void CalculateDistanceAllPoints(double * data_x_d, double * data_y_d, double * data_z_d, double * transformed_data_x_d, double * transformed_data_y_d, double * transformed_data_z_d, int * index_d, double * distance_d, int size_data)
{
int i = blockIdx.x*blockDim.x + threadIdx.x;

if(i < size_data)
{
int index = index_d[i];
distance_d[i] = sqrt(pow(data_x_d[index] - transformed_data_x_d[i],2) + pow(data_y_d[index] - transformed_data_y_d[i],2) + pow(data_z_d[index] - transformed_data_z_d[i],2));
}
}