#include "includes.h"
__global__ void CudaKernel_BatchResize_GRAY2GRAY( int src_width, unsigned char* src_image, int num_rects, int* rects, int dst_width, int dst_height, float* dst_ptr )
{
const int gid = blockIdx.x * blockDim.x + threadIdx.x;
const int dst_image_size = dst_width * dst_height;
if( num_rects*dst_image_size <= gid ){
return;
}

const int image_index = (int)(gid / dst_image_size);
const int pixel_index = gid % dst_image_size;

float scale_x = (float)(rects[image_index*4 + 2])/dst_width;
float fx = (float)(((pixel_index % dst_width)+0.5f)*scale_x - 0.5);
int coor_x_in_rect = floor(fx);
fx = 1.0f - (fx - (float)coor_x_in_rect);

float scale_y = (float)(rects[image_index*4 + 3])/dst_height;
float fy = (float)(((pixel_index / dst_width)+0.5f)*scale_y - 0.5);
int coor_y_in_rect = floor(fy);
fy = 1.0f - (fy - (float)coor_y_in_rect);

int src_x = rects[image_index*4 + 0];
int src_y = rects[image_index*4 + 1];

float value = 0.;
value += (float)src_image[src_width*(src_y + coor_y_in_rect + 0) + (src_x + coor_x_in_rect + 0)] * fx * fy;
value += (float)src_image[src_width*(src_y + coor_y_in_rect + 0) + (src_x + coor_x_in_rect + 1)] * (1.0f - fx)*fy;
value += (float)src_image[src_width*(src_y + coor_y_in_rect + 1) + (src_x + coor_x_in_rect + 0)] * fx*(1.0f - fy);
value += (float)src_image[src_width*(src_y + coor_y_in_rect + 1) + (src_x + coor_x_in_rect + 1)] * (1.0f - fx)*(1.0f - fy);

dst_ptr[blockIdx.x * blockDim.x + threadIdx.x] = value / 255.f;
}