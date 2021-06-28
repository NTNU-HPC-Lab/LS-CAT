#include "includes.h"
__global__ void lga_filtering_forward (const int n, const float *bottom_data, const float *filters, const int height, const int width, const int channel, const int radius, float *top_data){
int index = blockIdx.x * blockDim.x + threadIdx.x;
//    printf("OK\n");
//    printf("%d, %.2f, %.2f\n", index, bottom_data[index], top_data[index]);
if (index >= n)
{
return;
}
//    top_data[index]=1.0;
//    assert(0);
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
int shift = 0;
if (rr >= 0 && cc >= 0 && dd >= 0 && rr < height && cc < width
&& dd < channel)
shift = r * width + c + d * step;
int location =
(d + 1) * (wsize * wsize) + (r + radius) * wsize + c + radius;
top_data[index] +=
bottom_data[index + shift] * filters[fbase + location * step];
}
}
}
//        top_data[index]=1.0;
//        printf("%d, %d, %d, %.2f, %.2f\n", index, row, col, bottom_data[index], top_data[index]);
}