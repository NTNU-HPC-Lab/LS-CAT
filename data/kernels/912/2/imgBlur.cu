#include "includes.h"
__global__ void imgBlur(float* imgIn, float* imgOut, int imageWidth, int imageHeight)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int idy = blockIdx.y * blockDim.y + threadIdx.y;

if(idx<imageWidth && idy<imageHeight)
{
float sum = imgIn[idx*imageWidth+idy];

if(idx>0 && idy>0)
sum += imgIn[(idx-1)*imageWidth+(idy-1)];

if(idx>0)
sum += imgIn[(idx-1)*imageWidth+idy];

if(idx<imageWidth-1)
sum += imgIn[(idx+1)*imageWidth+idy];

if(idx<imageWidth-1 && idy<imageHeight-1)
sum += imgIn[(idx+1)*imageWidth+idy+1];

if(idx<imageWidth && idy>0)
sum += imgIn[(idx+1)*imageWidth+idy-1];

if(idy>0)
sum += imgIn[idx*imageWidth+idy-1];

if(idy<imageHeight)
sum += imgIn[idx*imageWidth+idy+1];

if(idx>0 && idy<imageHeight)
sum += imgIn[(idx-1)*imageWidth+idy+1];

imgOut[idx*imageWidth+idy] = sum / (float)(BLUR_SIZE*BLUR_SIZE);

}
}