#include "includes.h"
__global__ void process_coarseness_ek_pix(double * output_ak, double *output_ekh, double *output_ekv,int colsize, int rowsize,long lenOf_ek)
{
int y  = threadIdx.x + blockIdx.x * blockDim.x;
int x = threadIdx.y + blockIdx.y * blockDim.y;
double input1,input2;
int posx1 = x+lenOf_ek;
int posx2 = x-lenOf_ek;
int posy1 = y+lenOf_ek;
int posy2 = y-lenOf_ek;
if(y < (colsize) && x < (rowsize))
{
if(posx1 < (int)rowsize && posx2 >= 0)
{
input1 = output_ak[y * rowsize + posx1];
input2 = output_ak[y * rowsize + posx2];
output_ekh[y*rowsize+x] = fabs(input1 - input2);
}
else output_ekh[y*rowsize+x] = 0;

if(posy1 < (int)colsize && posy2 >= 0)
{
input1 = output_ak[posy1 * rowsize + x];
input2 = output_ak[posy2 * rowsize + x];
output_ekv[y*rowsize+x] = fabs(input1 - input2);
}
else output_ekv[y*rowsize+x] = 0;
}
}