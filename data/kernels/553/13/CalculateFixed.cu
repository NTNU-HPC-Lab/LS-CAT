#include "includes.h"
__global__ void CalculateFixed( const float *background, const float *target, const float *mask, float *fixed, const int wb, const int hb, const int wt, const int ht, const int oy, const int ox ){
const int dir[16][2] = {{-2, -2},				{0, -2},			{2, -2},
{-1, -1},	{0, -1},	{1, -1},
{-2, 0},	{-1, 0},				{1, 0},	{2, 0},
{-1, 1},	{0, 1},		{1, 1},
{-2, 2},				{0, 2},				{2, 2}};
const int coef[16] = {	1,		1,		1,
2,	2,	2,
1,	2,		2,	1,
2,	2,	2,
1,		1,		1};
const int num = 24;
const int yt = blockIdx.y * blockDim.y + threadIdx.y;
const int xt = blockIdx.x * blockDim.x + threadIdx.x;
const int curt = wt * yt + xt;
if (yt < ht && xt < wt && mask[curt] > 127.0f){
float sum[3] = {0}, bsum[3] = {0};
for (int i=0; i<16; i++){
int dxt = xt + dir[i][0];
int dyt = yt + dir[i][1];
int dcurt = wt * dyt + dxt;
int dxb = ox + dxt;
int dyb = oy + dyt;
if (dxt >= 0 && dxt < wt && dyt >= 0 && dyt < ht){
sum[0] += target[dcurt*3 + 0] * coef[i];
sum[1] += target[dcurt*3 + 1] * coef[i];
sum[2] += target[dcurt*3 + 2] * coef[i];
}
else {
sum[0] += target[curt*3 + 0] * coef[i];
sum[1] += target[curt*3 + 1] * coef[i];
sum[2] += target[curt*3 + 2] * coef[i];
}

if (dxt < 0 || dxt >= wt || dyt < 0 || dyt >= ht || mask[dcurt] < 127.0f){
dxb =	dxb <  0 ? 0:
dxb >= wb? wb-1: dxb;
dyb =	dyb <  0 ? 0:
dyb >= hb? hb-1: dyb;
int dcurb = wb * dyb + dxb;
bsum[0] += background[dcurb*3 + 0] * coef[i];
bsum[1] += background[dcurb*3 + 1] * coef[i];
bsum[2] += background[dcurb*3 + 2] * coef[i];
}
}
fixed[curt*3+0] = target[curt*3+0] - sum[0] / num + bsum[0] / num;
fixed[curt*3+1] = target[curt*3+1] - sum[1] / num + bsum[1] / num;
fixed[curt*3+2] = target[curt*3+2] - sum[2] / num + bsum[2] / num;
}
}