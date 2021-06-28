#include "includes.h"
__global__ void lga_data_backward (const int n, const float *filters, const float *top_diff, const int height, const int width, const int channel, const int radius, float *bottom_diff){
int index = blockIdx.x * blockDim.x + threadIdx.x;

if (index >= n)
{
return;
}
int step = height * width;
int wsize = 2 * radius + 1;
//      int fsize=wsize*wsize*3;
int fbase =
index / (step * channel) * (step * wsize * wsize * 3) + index % step;
int row = index % step / width;
int col = index % width;
int depth = index / step % channel;
for (int d = -1; d <= 1; d++)
{
for (int r = -radius; r <= radius; r++)
{
for (int c = -radius; c <= radius; c++)
{
int rr = r + row;
int cc = c + col;
int dd = d + depth;
//      int shift = 0;
if (rr >= 0 && cc >= 0 && dd >= 0 && rr < height && cc < width
&& dd < channel)
{
int shift = r * width + c + d * step;
//      int fshift= r*width+c;
int location =
(-d + 1) * (wsize * wsize) + (-r + radius) * wsize - c +
radius;
bottom_diff[index] +=
top_diff[index + shift] * filters[fbase + r * width + c +
location * step];
}
else
{
int location =
(d + 1) * (wsize * wsize) + (r + radius) * wsize + c +
radius;
bottom_diff[index] +=
top_diff[index] * filters[fbase + location * step];
}
}
}
}
}