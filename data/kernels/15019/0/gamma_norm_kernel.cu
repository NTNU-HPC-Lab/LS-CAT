#include "includes.h"
__global__ void gamma_norm_kernel(float* img, int image_height, int image_width, int image_step)
{
// The thread block has size (3,n). The first dimension of the thread block
// corresponds to color channels.
int channel = threadIdx.x;
// The columns of the image are mapped to the first dimension of the block
// grid, but to the second dimension of the thread block, as the first
// already corresponds to color channels.
int pixel_x = blockIdx.x * blockDim.y + threadIdx.y;
// If current position is outside the image, stop here
if(pixel_x >= image_width)
{
return;
}
// The columns of the image are mapped to the second dimension of the block
// grid, but to the third dimension of the thread block.
int pixel_y = blockIdx.y * blockDim.z + threadIdx.z;
// If current position is outside the image, stop here
if(pixel_y >= image_height)
{
return;
}

// Each row has image_step pixels and each pixel has three channels
int in_pixel_idx = pixel_y * image_step + pixel_x * 3 + channel;

// Finally perform the normalization
img[in_pixel_idx] = sqrt(img[in_pixel_idx] / 256.0f);

}