#include "includes.h"
__global__ void sga_down_data_backward (const int n, const float *filters, float *top_diff, const int height, const int width, const int depth, const int wsize, float *bottom_diff){
int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index >= n)
{
return;
}
int step = height * width;
int base = index / width * step * depth + index % width;	//up->down
int fbase = index / width * step * wsize + index % width;
for (int row = height - 1; row >= 0; row--)
{
int shift = fbase + row * width;
for (int d = 0; d < depth; d++)
{
int location = base + d * step + row * width;
float temp = top_diff[location];
if (row + 1 < height)
temp +=
top_diff[location + width] * filters[shift + width + step];
if (row + 2 < height)
temp +=
top_diff[location + 2 * width] * filters[shift + 2 * width +
2 * step];
if (row + 1 < height && d + 1 < depth)
temp +=
top_diff[location + width + step] * filters[shift + width +
3 * step];
if (row + 1 < height && d - 1 >= 0)
temp +=
top_diff[location + width - step] * filters[shift + width +
4 * step];
top_diff[location] = temp;
bottom_diff[location] += temp * filters[shift];
}
}
/*	for(int d = 0; d < depth; d ++){
int shift = fbase;
int location = base + d * step;
bottom_diff[location] += top_diff[location] * (filters[shift + step] + filters[shift + 2*step] + filters[shift + 3*step] + filters[shift + 4*step]);
//       bottom_diff[location] += top_diff[location];
shift += width;
location += width;
bottom_diff[location] += top_diff[location] * filters[shift + 2*step];
}
for(int row=1;row<height;row++){
int location = base + row * width;
int shift = fbase + row * width;
bottom_diff[location] += top_diff[location] * filters[shift + 3*step];
location += (depth - 1)*step;
bottom_diff[location] += top_diff[location] * filters[shift + 4*step];
}
*/
for (int row = 0; row < height; row++)
{
int location = base + row * width;
int shift = fbase + row * width;
bottom_diff[location] += top_diff[location] * filters[shift + 3 * step];
location += (depth - 1) * step;
bottom_diff[location] += top_diff[location] * filters[shift + 4 * step];
}
}