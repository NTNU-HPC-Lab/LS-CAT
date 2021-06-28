#include "includes.h"
__global__ void myTextureKernel(cudaSurfaceObject_t SurfObj, size_t width, size_t height)
{
for (int idy = blockIdx.y * blockDim.y + threadIdx.y;
idy < height;
idy += blockDim.y * gridDim.y)
{
for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
idx < width;
idx += blockDim.x * gridDim.x)
{
uchar4 data = make_uchar4(255,255,255,255);
// Read from input surface
//surf2Dread(&data,  inputSurfObj, x * sizeof(uchar4), y);
// Write to output surface
surf2Dwrite(data, SurfObj, idx * sizeof(uchar4), idy);
}
}
}