#include "includes.h"
__global__ void processBoxes(int size, const float* src, float* dst,const int stridex, const int stridey)
{
int index = blockIdx.x * blockDim.x + threadIdx.x;
if(index < size)
{
float4* src_boxes = (float4*)src + index;
float4* dst_boxes = (float4*)dst + index;
float4 boxes = *src_boxes;
float4 new_boxes = {0};
new_boxes.x = boxes.x - boxes.z * stridex / 2;
new_boxes.y = boxes.y - boxes.w * stridey / 2;
new_boxes.z = boxes.x + boxes.z * stridex / 2;
new_boxes.w = boxes.y + boxes.w * stridey / 2;
*dst_boxes = new_boxes;
}
}