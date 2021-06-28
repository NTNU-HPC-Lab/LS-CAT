#include "includes.h"


__global__ void kRgb2XYZ(uchar4* inputImg, float4* outputImg, int width, int height)
{
int offsetBlock = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * width;
int offset = offsetBlock + threadIdx.x + threadIdx.y * width;

uchar4 nPixel=inputImg[offset];

float _b=(float)nPixel.x/255.0;
float _g=(float)nPixel.y/255.0;
float _r=(float)nPixel.z/255.0;

float x=_r*0.412453	+_g*0.357580	+_b*0.180423;
float y=_r*0.212671	+_g*0.715160	+_b*0.072169;
float z=_r*0.019334	+_g*0.119193	+_b*0.950227;

float4 fPixel;
fPixel.x=x;
fPixel.y=y;
fPixel.z=z;

outputImg[offset]=fPixel;
}