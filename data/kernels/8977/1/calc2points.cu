#include "includes.h"



__global__ void calc2points(float* point_coordinate_1, float* point_coordinate_2 , float* coordinates_arr)
{
int tid = threadIdx.x; // 52

coordinates_arr[tid] = pow(point_coordinate_1[tid] - point_coordinate_2[tid],2);
}