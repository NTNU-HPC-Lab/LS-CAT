#include "includes.h"
__global__ void BilinearResampleKernel(float *input, float *output, int inputWidth, int inputHeight, int outputWidth, int outputHeight)
{
int id = blockDim.x * blockIdx.y * gridDim.x
+ blockDim.x * blockIdx.x
+ threadIdx.x;
int size = outputWidth * outputHeight;
float iT, iB;

if (id < size)
{
//output point coordinates
int px = id % outputWidth;
int py = id / outputWidth;

float xRatio = (float)(inputWidth - 1) / (outputWidth - 1);
float yRatio = (float)(inputHeight - 1) / (outputHeight - 1);

//corresponding coordinates in the original image
float x = xRatio * px;
float y = yRatio * py;

//corresponding integer (pixel) coordinates in the original image
int xL = (int)floor(x);
int xR = (int)ceil(x);
int yT = (int)floor(y);
int yB = (int)ceil(y);


//inverse distances to these points
float dL = 1.0f - (x - xL);
float dR = 1.0f - (xR - x);
float dT = 1.0f - (y - yT);
float dB = 1.0f - (yB - y);

//values at those points
float topLeft = input[yT * inputWidth + xL];
float topRight = input[yT * inputWidth + xR];
float bottomLeft = input[yB * inputWidth + xL];
float bottomRight = input[yB * inputWidth + xR];

//linear interpolation in X (i.e., top and bottom pairs of points)
if (xL == xR) { //interpolated points corresponds exactly to one integer x-coordinate in the original image, choose any one of them
iT = topLeft;
iB = bottomLeft;
}
else {
iT = topLeft * dL + topRight * dR;
iB = bottomLeft * dL + bottomRight * dR;
}

//linear interpolation in Y (i.e., linear interpolation of those two points)
if (yT == yB) //interpolated points corresponds exactly to one integer ycoordinate in the original image, choose any one of them
{
output[py * outputWidth + px] = iT;
}
else {
output[py * outputWidth + px] = iT * dT + iB * dB;
}
}
}