#include "includes.h"
__global__ void cuda_copyRegion(unsigned char *dst, unsigned char *src,int stepDst, int stepSrc, int dst_width, int dst_height, int src_width, int src_height, int dst_xoffset, int dst_yoffset, int dst_widthToCrop, int dst_heightToCrop, int src_xoffset, int src_yoffset, int src_widthToCrop, int src_heightToCrop, int numChannel)
{
//    printf("stepSrc - Dst = %d - %d\n", stepSrc, stepDst);
int col = threadIdx.x + blockIdx.x * blockDim.x;
int row = threadIdx.y + blockIdx.y * blockDim.y;

int dst_col = col + dst_xoffset;
int dst_row = row + dst_yoffset;

int src_col = col + src_xoffset;
int src_row = row + src_yoffset;


if(row < dst_heightToCrop && col < dst_widthToCrop && dst_col < dst_width&& dst_row < dst_height)
{
if(numChannel==1)
{
dst[dst_row * (stepDst) + dst_col] = src[src_row * (stepSrc) + src_col];
}
if(numChannel==3)
{
int dst_step = dst_row * (stepDst) + dst_col;
int src_step = src_row * (stepSrc) + src_col;
dst[3 * dst_step] = src[3 * src_step];
dst[3 * dst_step + 1] = src[3 * src_step + 1];
dst[3 * dst_step + 2] = src[3 * src_step + 2];
}
}
}