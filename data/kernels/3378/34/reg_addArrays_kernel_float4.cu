#include "includes.h"
__global__ void reg_addArrays_kernel_float4(float4 *array1_d, float4 *array2_d)
{
const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
if(tid < c_VoxelNumber){
float4 a = array1_d[tid];
float4 b = array1_d[tid];
array1_d[tid] = make_float4(a.x+b.x,a.y+b.y,a.z+b.z,a.w+b.w);
}
}