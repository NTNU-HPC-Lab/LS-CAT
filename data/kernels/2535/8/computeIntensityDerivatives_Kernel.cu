#include "includes.h"



#define T_PER_BLOCK 16
#define MINF __int_as_float(0xff800000)




__global__ void computeIntensityDerivatives_Kernel(float2* d_output, const float* d_input, unsigned int width, unsigned int height)
{
const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

if (x < width && y < height)
{
d_output[y*width + x] = make_float2(MINF, MINF);

//derivative
if (x > 0 && x < width - 1 && y > 0 && y < height - 1)
{
float pos00 = d_input[(y - 1)*width + (x - 1)]; if (pos00 == MINF) return;
float pos01 = d_input[(y - 0)*width + (x - 1)];	if (pos01 == MINF) return;
float pos02 = d_input[(y + 1)*width + (x - 1)];	if (pos02 == MINF) return;

float pos10 = d_input[(y - 1)*width + (x - 0)]; if (pos10 == MINF) return;
//float pos11 = d_input[(y-0)*width + (x-0)]; if (pos11 == MINF) return;
float pos12 = d_input[(y + 1)*width + (x - 0)]; if (pos12 == MINF) return;

float pos20 = d_input[(y - 1)*width + (x + 1)]; if (pos20 == MINF) return;
float pos21 = d_input[(y - 0)*width + (x + 1)]; if (pos21 == MINF) return;
float pos22 = d_input[(y + 1)*width + (x + 1)]; if (pos22 == MINF) return;

float resU = (-1.0f)*pos00 + (1.0f)*pos20 +
(-2.0f)*pos01 + (2.0f)*pos21 +
(-1.0f)*pos02 + (1.0f)*pos22;
resU /= 8.0f;

float resV = (-1.0f)*pos00 + (-2.0f)*pos10 + (-1.0f)*pos20 +
(1.0f)*pos02 + (2.0f)*pos12 + (1.0f)*pos22;
resV /= 8.0f;

d_output[y*width + x] = make_float2(resU, resV);
}
}
}