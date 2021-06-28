#include "includes.h"
__global__ void sga_right_forward (const int n, const float *filters, const int height, const int width, const int depth, const int wsize, float *top_data){

int index = blockIdx.x * blockDim.x + threadIdx.x;

if (index >= n)
{
return;
}
int step = height * width;
//   int wsize=radius+1;

int base = index / height * step * depth + (index % height) * width;	//up->down
int fbase = index / height * step * wsize + (index % height) * width;

for (int col = 0; col < width; col++)
{
int shift = fbase + col;
for (int d = 0; d < depth; d++)
{
float temp = 0;
int location = base + d * step + col;
temp += top_data[location] * filters[shift];
if (col - 1 >= 0)
temp += top_data[location - 1] * filters[shift + step];
else
temp += top_data[location] * filters[shift + step];
if (col - 2 >= 0)
temp += top_data[location - 2] * filters[shift + 2 * step];
else
temp += top_data[location] * filters[shift + 2 * step];
if (col - 1 >= 0 && d - 1 >= 0)
temp += top_data[location - 1 - step] * filters[shift + 3 * step];
else
temp += top_data[location] * filters[shift + 3 * step];
if (col - 1 >= 0 && d + 1 < depth)
temp += top_data[location - 1 + step] * filters[shift + 4 * step];
else
temp += top_data[location] * filters[shift + 4 * step];

top_data[location] = temp;
}
}
}