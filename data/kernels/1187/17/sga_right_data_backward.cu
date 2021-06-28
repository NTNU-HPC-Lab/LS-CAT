#include "includes.h"
__global__ void sga_right_data_backward (const int n, const float *filters, float *top_diff, const int height, const int width, const int depth, const int wsize, float *bottom_diff){
int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index >= n)
{
return;
}
int step = height * width;
int base = index / height * step * depth + (index % height) * width;	//up->down
int fbase = index / height * step * wsize + (index % height) * width;

for (int col = width - 1; col >= 0; col--)
{
int shift = fbase + col;
for (int d = 0; d < depth; d++)
{
int location = base + d * step + col;
float temp = top_diff[location];
if (col + 1 < width)
temp += top_diff[location + 1] * filters[shift + 1 + step];
if (col + 2 < width)
temp += top_diff[location + 2] * filters[shift + 2 + 2 * step];
if (col + 1 < width && d + 1 < depth)
temp +=
top_diff[location + 1 + step] * filters[shift + 1 + 3 * step];
if (col + 1 < width && d - 1 >= 0)
temp +=
top_diff[location + 1 - step] * filters[shift + 1 + 4 * step];
top_diff[location] = temp;
bottom_diff[location] += (temp * filters[shift]);
}
}
/*
for(int d = 0; d < depth; d ++){
int shift = fbase;// + width*(height-1);
int location = base;// + width*(height-1) + d * step;
bottom_diff[location] += top_diff[location] * (filters[shift + step] + filters[shift + 2*step] + filters[shift + 3*step] + filters[shift + 4*step]);
//       bottom_diff[location] += top_diff[location];
shift += 1;
location += 1;
bottom_diff[location] += top_diff[location] * filters[shift + 2*step];
}
for(int col=1;col<width;col++){
int shift = fbase + col;
int location = base + col;
bottom_diff[location] += top_diff[location] * filters[shift + 3*step];
location += (depth - 1)*step;
bottom_diff[location] += top_diff[location] * filters[shift + 4*step];
}*/
for (int col = 0; col < width; col++)
{
int shift = fbase + col;
int location = base + col;
bottom_diff[location] += top_diff[location] * filters[shift + 3 * step];
location += (depth - 1) * step;
bottom_diff[location] += top_diff[location] * filters[shift + 4 * step];
}
}