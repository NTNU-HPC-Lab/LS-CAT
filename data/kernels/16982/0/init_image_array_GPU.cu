#include "includes.h"
__global__ void init_image_array_GPU(unsigned long long int* image, int pixels_per_image)
{
int my_pixel = threadIdx.x + blockIdx.x*blockDim.x;
if (my_pixel < pixels_per_image)
{
// -- Set the current pixel to 0 and return, avoiding overflow when more threads than pixels are used:
image[my_pixel] = (unsigned long long int)(0);    // Initialize non-scatter image
my_pixel += pixels_per_image;                     //  (advance to next image)
image[my_pixel] = (unsigned long long int)(0);    // Initialize Compton image
my_pixel += pixels_per_image;                     //  (advance to next image)
image[my_pixel] = (unsigned long long int)(0);    // Initialize Rayleigh image
my_pixel += pixels_per_image;                     //  (advance to next image)
image[my_pixel] = (unsigned long long int)(0);    // Initialize multi-scatter image
}
}