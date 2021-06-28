#include "includes.h"
__global__ void d_updateTransforms (float* d_currentTransform, float3* d_cameraPosition)
{
d_cameraPosition->x = d_currentTransform[3];
d_cameraPosition->y = d_currentTransform[7];
d_cameraPosition->z = d_currentTransform[11];
}