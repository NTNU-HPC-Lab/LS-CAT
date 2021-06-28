#include "includes.h"
__device__ float gpu_applyFilter(float *image, int stride, float *matrix, int filter_dim)
{
////////////////
// TO-DO #5.2 ////////////////////////////////////////////////
// Implement the GPU version of cpu_applyFilter()           //
//                                                          //
// Does it make sense to have a separate gpu_applyFilter()? //
//////////////////////////////////////////////////////////////

return 0.0f;
}
__global__ void gpu_gaussian(int width, int height, float *image, float *image_out)
{
float gaussian[9] = { 1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f };

int index_x = blockIdx.x * blockDim.x + threadIdx.x;
int index_y = blockIdx.y * blockDim.y + threadIdx.y;

if (index_x < (width - 2) && index_y < (height - 2))
{
int offset_t = index_y * width + index_x;
int offset   = (index_y + 1) * width + (index_x + 1);

image_out[offset] = gpu_applyFilter(&image[offset_t],
width, gaussian, 3);
}
}