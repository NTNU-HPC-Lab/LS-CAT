#include "includes.h"
__global__ void reg_addArrays_kernel_float(float *array1_d, float *array2_d)
{
const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
if(tid < c_VoxelNumber){
array1_d[tid] += array2_d[tid];
}
}