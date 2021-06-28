#include "includes.h"
__global__	void	BuildColorFieldDev(float* data, uchar4* colors, float* minmax, uint xx, uint yy)
{
float	mn = minmax[0];
float	mx = minmax[1];

float	median = (mx - mn)/2.0f;

const uint idx = threadIdx.x*gridDim.x/yy/yy + blockIdx.x/xx;

float	val = data[idx];

uchar4	col;

#if	1

if(val < median)
{
float alpha = (val - mn)/(median - mn);

col.x = 0;
col.y = 255*(1-alpha);
col.z = 255*alpha;

}else
{
float alpha = (val - median)/(mx - median);

col.x = 255*alpha;
col.y = 0;
col.z = 255*(1-alpha);
}
#else

float	alpha = 1;

if(!(val < 0.1 || mn == mx || mx < 0.1))
alpha = val/(mx-mn);

col.x = 255*(1-alpha);
col.y = 255*(1-alpha);
col.z = 255*(1-alpha);

#endif
col.w = 255;

const	uint	col_idx = threadIdx.x*gridDim.x + blockIdx.x;

colors[col_idx] = col;

}