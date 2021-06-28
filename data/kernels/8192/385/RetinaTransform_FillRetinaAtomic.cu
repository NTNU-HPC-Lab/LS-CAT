#include "includes.h"
__device__ void EstimateParForSubsample(float* subImageDefs, bool safeBounds, int inputWidth, int inputHeight, int2 & subImg, int & diameterPix)
{
diameterPix = (int)( fminf( (float)inputWidth,(float)inputHeight ) * subImageDefs[2] ); // <0,1>

subImg.x = (int)((float)inputWidth * (subImageDefs[0] + 1) * 0.5f) ;//- diameterPix / 2;
subImg.y = (int)((float)inputHeight * (subImageDefs[1] + 1) * 0.5f);// - diameterPix / 2;

int maxDiameter = min(inputWidth - 1, inputHeight - 1);

diameterPix = max(1, diameterPix);
diameterPix = min(maxDiameter, diameterPix);

if (safeBounds)
{
subImg.x = max(subImg.x, 1);
subImg.y = max(subImg.y, 1);
subImg.x = min(subImg.x, inputWidth - diameterPix - 1);
subImg.y = min(subImg.y, inputHeight - diameterPix - 1);
}
}
__global__ void RetinaTransform_FillRetinaAtomic (float * subImageDefs, float* input, int inputWidth, int inputHeight, float* output,int outputDataSize, float* retinaMask, int retinaDataSize, int retinaMaskColHint, float* retinaDataInserted)
{
int id_pxl = blockDim.x * blockIdx.y * gridDim.x
+ blockDim.x * blockIdx.x
+ threadIdx.x;

int2 subImg;
int diameterPix;
bool  safeBounds = 0;

int x = id_pxl % inputWidth;
int y = id_pxl/inputWidth;

EstimateParForSubsample( subImageDefs,  safeBounds, inputWidth,  inputHeight,  subImg, diameterPix );

if (id_pxl<inputWidth*inputHeight)
{
float minDist = 999999.9; // ??>? should be written bette
int minIdx = 1;
for (int id_retinaPoint=0 ; id_retinaPoint<retinaDataSize ; id_retinaPoint++)
{
float x_mask = (retinaMask[id_retinaPoint*retinaMaskColHint]*diameterPix);
float y_mask = (retinaMask[id_retinaPoint*retinaMaskColHint+1]*diameterPix);

x_mask += subImg.x;
y_mask += subImg.y;

float dist = (x-x_mask)*(x-x_mask) + (y-y_mask)*(y-y_mask);

if (dist<minDist)
{
minDist = dist;
minIdx  = id_retinaPoint;
}
}
atomicAdd(output + minIdx , input[id_pxl]);
atomicAdd(retinaDataInserted + minIdx , 1);
}
}