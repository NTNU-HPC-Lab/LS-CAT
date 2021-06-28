#include "includes.h"
__global__ void getOptimalShifts( float2 * __restrict__ optimalShifts, const float2 * __restrict__ bestShifts, int imageCount, int tileCountX, int tileCountY, int optimalShiftsPitch, int referenceImage, int imageToTrack)
{
int tileIdxX = blockIdx.x * blockDim.x + threadIdx.x;
int tileIdxY = blockIdx.y * blockDim.y + threadIdx.y;

if (tileIdxX >= tileCountX || tileIdxY >= tileCountY)
return;

int n1 = imageCount - 1;

const float2* r = &bestShifts[(tileIdxX + tileIdxY * tileCountX) * n1];

float2 totalShift = make_float2(0, 0);
if (referenceImage < imageToTrack)
{
for (int i = referenceImage; i < imageToTrack; i++)
{
totalShift.x += r[i].x;
totalShift.y += r[i].y;
}
}
else if(imageToTrack < referenceImage)
{
for (int i = imageToTrack; i < referenceImage; i++)
{
totalShift.x -= r[i].x;
totalShift.y -= r[i].y;
}
}

*(((float2*)((char*)(optimalShifts) +optimalShiftsPitch * tileIdxY)) + tileIdxX) = totalShift;
}