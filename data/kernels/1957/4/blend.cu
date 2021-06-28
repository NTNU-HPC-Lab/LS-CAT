#include "includes.h"
__global__ void blend(float *cmap, float* oldd, float* newd, float weight,int * params)
{
int ax = blockIdx.x*blockDim.x + threadIdx.x;
int ay = blockIdx.y*blockDim.y + threadIdx.y;

int ch = params[0];
int ah = params[1];
int aw = params[2];

int slice_a = ah * aw;
int pitch_a = aw;

float thre = 0.05;

if (ax < aw&& ay < ah)
{
float fa = cmap[ay*pitch_a + ax];
if (fa < thre)
fa = 0.0f;
else fa = weight;
for (int i = 0; i < ch; i++)
{

newd[i*slice_a + ay*pitch_a + ax] = oldd[i*slice_a + ay*pitch_a + ax]* fa + newd[i*slice_a + ay*pitch_a + ax] * (1.0-fa);
}
}
}