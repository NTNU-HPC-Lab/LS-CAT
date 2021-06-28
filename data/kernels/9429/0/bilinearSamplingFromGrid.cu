#include "includes.h"
// Bilinear sampling is done in BHWD (coalescing is not obvious in BDHW)
// we assume BHWD format in inputImages
// we assume BHW(YX) format on grids

__device__ bool between(int value, int lowerBound, int upperBound)
{
return (value >= lowerBound && value <= upperBound);
}
__device__ void getTopLeft(float x, int width, int& point, float& weight)
{
/* for interpolation :
stores in point and weight :
- the x-coordinate of the pixel on the left (or y-coordinate of the upper pixel)
- the weight for interpolating
*/

float xcoord = (x + 1) * (width - 1) / 2;
point = floor(xcoord);
weight = 1 - (xcoord - point);
}
__global__ void bilinearSamplingFromGrid(float* inputImages_data, int inputImages_strideBatch, int inputImages_strideChannels, int inputImages_strideHeight, int inputImages_strideWidth, float* grids_data, int grids_strideBatch, int grids_strideYX, int grids_strideHeight, int grids_strideWidth, float* output_data, int output_strideBatch, int output_strideChannels, int output_strideHeight, int output_strideWidth, int inputImages_channels, int inputImages_height, int inputImages_width, int output_width)
{
// each (32,16) block 16 output pixels (for coalescing the grid read)
// x,y = coordinates (xOut = blockIdx.x*16+blockDim.y+threadIdx.y)
// z = batch index
// threadIdx.x : used for features (coalescing is trivial)

const int xOut = blockIdx.x*blockDim.y+threadIdx.y;
const bool withinImageBounds = xOut < output_width;
const bool withinGridBounds = blockIdx.x*blockDim.y + threadIdx.x / 2 < output_width;
const int yOut = blockIdx.y;
const int width = inputImages_width;
const int height = inputImages_height;

const int b = blockIdx.z;

float yf,xf;

__shared__ float gridData[32];
if (threadIdx.y==0 && withinGridBounds)
{
gridData[threadIdx.x] = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth + threadIdx.x];
}
__syncthreads();
if(!withinImageBounds) return;
yf = gridData[threadIdx.y*2];
xf = gridData[threadIdx.y*2+1];

int yInTopLeft, xInTopLeft;
float yWeightTopLeft, xWeightTopLeft;
getTopLeft(xf, inputImages_width, xInTopLeft, xWeightTopLeft);
getTopLeft(yf, inputImages_height, yInTopLeft, yWeightTopLeft);

const int outAddress = output_strideBatch * b + output_strideHeight * yOut + output_strideWidth * xOut;
const int inTopLeftAddress = inputImages_strideBatch * b + inputImages_strideHeight * yInTopLeft + inputImages_strideWidth * xInTopLeft;
const int inTopRightAddress = inTopLeftAddress + inputImages_strideWidth;
const int inBottomLeftAddress = inTopLeftAddress + inputImages_strideHeight;
const int inBottomRightAddress = inBottomLeftAddress + inputImages_strideWidth;

float v=0;
float inTopLeft=0;
float inTopRight=0;
float inBottomLeft=0;
float inBottomRight=0;

bool topLeftIsIn = between(xInTopLeft, 0, width-1) && between(yInTopLeft, 0, height-1);
bool topRightIsIn = between(xInTopLeft+1, 0, width-1) && between(yInTopLeft, 0, height-1);
bool bottomLeftIsIn = between(xInTopLeft, 0, width-1) && between(yInTopLeft+1, 0, height-1);
bool bottomRightIsIn = between(xInTopLeft+1, 0, width-1) && between(yInTopLeft+1, 0, height-1);

// interpolation happens here
for(int t=threadIdx.x; t<inputImages_channels; t+= blockDim.x)
{
if(topLeftIsIn) inTopLeft = inputImages_data[inTopLeftAddress + t];
if(topRightIsIn) inTopRight = inputImages_data[inTopRightAddress + t];
if(bottomLeftIsIn) inBottomLeft = inputImages_data[inBottomLeftAddress + t];
if(bottomRightIsIn) inBottomRight = inputImages_data[inBottomRightAddress + t];

v = xWeightTopLeft * yWeightTopLeft * inTopLeft
+ (1 - xWeightTopLeft) * yWeightTopLeft * inTopRight
+ xWeightTopLeft * (1 - yWeightTopLeft) * inBottomLeft
+ (1 - xWeightTopLeft) * (1 - yWeightTopLeft) * inBottomRight;

output_data[outAddress + t] = v;
}
}