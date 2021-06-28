#include "includes.h"
__global__ void MotionVec(float *new_image_dev, float *old_image_dev, uchar4 *Image_dev, int w, int h )
{
const int ix = blockDim.x * blockIdx.x + threadIdx.x;
const int iy = blockDim.y * blockIdx.y + threadIdx.y;
const float x = (float)ix + 0.5f;
const float y = (float)iy + 0.5f;
float diff = 0;

diff = old_image_dev[w*iy + ix] - new_image_dev[w*iy + ix];
diff *= diff;

float threshold = 5000;

if (diff > threshold)
{
Image_dev[w*iy + ix].x = 0;			//B  /* MODIFY CODE HERE*/
Image_dev[w*iy + ix].y = 0;			//G  /* MODIFY CODE HERE*/
Image_dev[w*iy + ix].z = 255;		//R  /* MODIFY CODE HERE*/
}
}