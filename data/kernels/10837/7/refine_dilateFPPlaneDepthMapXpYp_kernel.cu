#include "includes.h"
__global__ void refine_dilateFPPlaneDepthMapXpYp_kernel(float* fpPlaneDepthMap, int fpPlaneDepthMap_p, float* maskMap, int maskMap_p, int width, int height, int xp, int yp, float fpPlaneDepth)
{
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;

if((x + xp >= 0) && (y + yp >= 0) && (x + xp < width) && (y + yp < height) && (x < width) && (y < height))
{
float depth = maskMap[y * maskMap_p + x];
if(depth > 0.0f)
{
fpPlaneDepthMap[(y + yp) * fpPlaneDepthMap_p + (x + xp)] = fpPlaneDepth;
};
};
}