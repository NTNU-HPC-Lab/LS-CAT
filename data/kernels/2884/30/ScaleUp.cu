#include "includes.h"
__global__ void ScaleUp(float *d_Result, float *d_Data, int width, int pitch, int height, int newpitch)
{
const int tx = threadIdx.x;
const int ty = threadIdx.y;
int x = blockIdx.x*SCALEUP_W + 2*tx;
int y = blockIdx.y*SCALEUP_H + 2*ty;
if (x<2*width && y<2*height) {
int xl = blockIdx.x*(SCALEUP_W/2) + tx;
int yu = blockIdx.y*(SCALEUP_H/2) + ty;
int xr = min(xl + 1, width - 1);
int yd = min(yu + 1, height - 1);
float vul = d_Data[yu*pitch + xl];
float vur = d_Data[yu*pitch + xr];
float vdl = d_Data[yd*pitch + xl];
float vdr = d_Data[yd*pitch + xr];
d_Result[(y + 0)*newpitch + x + 0] = vul;
d_Result[(y + 0)*newpitch + x + 1] = 0.50f*(vul + vur);
d_Result[(y + 1)*newpitch + x + 0] = 0.50f*(vul + vdl);
d_Result[(y + 1)*newpitch + x + 1] = 0.25f*(vul + vur + vdl + vdr);
}
}