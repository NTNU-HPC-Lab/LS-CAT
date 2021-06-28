#include "includes.h"
__global__ void downSanple420_gpu(cudaTextureObject_t ch1, cudaTextureObject_t ch2, uint8_t *downCh1, uint8_t *downCh2, size_t width, size_t height)
{
int2 threadCoord = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
if (threadCoord.x < width && (threadCoord.y << 1) < height)
{
int2 pixelCoord;
cudaTextureObject_t *ch;
uint8_t *downCh;

// Remember thread divergence happens at the wrap level only, that will parallelize well
if (threadCoord.x < (width >> 1))
{
pixelCoord = make_int2(threadCoord.x << 1, threadCoord.y << 1);
ch = &ch1;
downCh = downCh1;
}
else
{
pixelCoord = make_int2((threadCoord.x - (width >> 1)) << 1, threadCoord.y << 1);
ch = &ch2;
downCh = downCh2;
}

int16_t bias = (pixelCoord.x & 1) + 1;
uint16_t pixel = (tex2D<uint16_t>(*ch, pixelCoord.x, pixelCoord.y) + tex2D<uint16_t>(*ch, pixelCoord.x + 1, pixelCoord.y)
+ tex2D<uint16_t>(*ch, pixelCoord.x, pixelCoord.y + 1) + tex2D<uint16_t>(*ch, pixelCoord.x + 1, pixelCoord.y + 1) + bias) >> 2;
downCh[((pixelCoord.y >> 1) * width + pixelCoord.x) >> 1] = (uint8_t)pixel;
}
}