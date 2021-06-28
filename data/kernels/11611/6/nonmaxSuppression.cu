#include "includes.h"
__device__ int tex_i(const int * ptData,int y,int x,int step)
{
return ptData[y*step+x];
}
__global__ void nonmaxSuppression(const short2* kpLoc_Device, int count, const int* score_DeviceMat,int cols,int rows,short2* locFinal, float* responseFinal)
{

const int kpIdx = threadIdx.x + blockIdx.x * blockDim.x;

if (kpIdx < count)
{
short2 loc = kpLoc_Device[kpIdx];

int score_Device = tex_i( score_DeviceMat,loc.y, loc.x,cols);

bool ismax =
score_Device > tex_i( score_DeviceMat,loc.y - 1, loc.x - 1,cols) &&
score_Device > tex_i( score_DeviceMat,loc.y - 1, loc.x    ,cols) &&
score_Device > tex_i( score_DeviceMat,loc.y - 1, loc.x + 1,cols) &&

score_Device > tex_i( score_DeviceMat,loc.y    , loc.x - 1,cols) &&
score_Device > tex_i( score_DeviceMat,loc.y    , loc.x + 1,cols) &&

score_Device > tex_i( score_DeviceMat,loc.y + 1, loc.x - 1,cols) &&
score_Device > tex_i( score_DeviceMat,loc.y + 1, loc.x    ,cols) &&
score_Device > tex_i( score_DeviceMat,loc.y + 1, loc.x + 1,cols);

if (ismax)
{
const unsigned int ind = atomicInc(&g_counter, (unsigned int)(-1));

locFinal[ind] = loc;
responseFinal[ind] = static_cast<float>(score_Device);
}
}

}