#include "includes.h"
__global__ void sga_up_forward (const int n, const float *filters, const int height, const int width, const int depth, const int wsize, float *top_data){

int index = blockIdx.x * blockDim.x + threadIdx.x;

if (index >= n)
{
return;
}
int step = height * width;
//   int wsize=radius+1;

int base = index / width * step * depth + index % width;	//up->down
int fbase = index / width * step * wsize + index % width;

for (int row = height - 1; row >= 0; row--)
{
int shift = fbase + row * width;
for (int d = 0; d < depth; d++)
{
float temp = 0;
int location = base + d * step + row * width;
temp += top_data[location] * filters[shift];
if (row + 1 < height)
temp += top_data[location + width] * filters[shift + step];
else
temp += top_data[location] * filters[shift + step];
if (row + 2 < height)
temp +=
top_data[location + 2 * width] * filters[shift + 2 * step];
else
temp += top_data[location] * filters[shift + 2 * step];
if (row + 1 < height && d - 1 >= 0)
temp +=
top_data[location + width - step] * filters[shift + 3 * step];
else
temp += top_data[location] * filters[shift + 3 * step];
if (row + 1 < height && d + 1 < depth)
temp +=
top_data[location + width + step] * filters[shift + 4 * step];
else
temp += top_data[location] * filters[shift + 4 * step];

top_data[location] = temp;

}
}
}