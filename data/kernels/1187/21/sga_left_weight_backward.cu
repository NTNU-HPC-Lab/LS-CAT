#include "includes.h"
__global__ void sga_left_weight_backward (const int n, const float *bottom_data, const float *top_data, const float *temp_diff, const int height, const int width, const int depth, const int wsize, float *filters_diff){

int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index >= n)
{
return;
}
int step = height * width;
int base = index / step * step * depth + index % step;	//up->down
int fbase = index / step * step * wsize + index % step;

//   int row = index%step/width;
int col = index % step % width;
for (int i = 0; i < depth; i++)
filters_diff[fbase] +=
temp_diff[base + i * step] * bottom_data[base + i * step];
if (col + 1 < width)
{
int location = fbase + step;
for (int i = 0; i < depth; i++)
filters_diff[location] +=
temp_diff[base + i * step] * top_data[base + i * step + 1];

location = fbase + 3 * step;
filters_diff[location] += temp_diff[base] * bottom_data[base];
for (int i = 1; i < depth; i++)
filters_diff[location] +=
temp_diff[base + i * step] * top_data[base + (i - 1) * step + 1];

location = fbase + 4 * step;
filters_diff[location] +=
temp_diff[base + (depth - 1) * step] * bottom_data[base + (depth - 1) * step];
for (int i = 0; i < depth - 1; i++)
filters_diff[location] +=
temp_diff[base + i * step] * top_data[base + (i + 1) * step + 1];
}
/*
else{
//int location = fbase + step;
for(int i=0; i<depth; i++){
float temp = temp_diff[base+i*step]*bottom_data[base+i*step];
filters_diff[fbase + step] += temp; //temp_diff[base+i*step]*bottom_data[base+i*step];
filters_diff[fbase + 3*step] += temp; //temp_diff[base+i*step]*bottom_data[base+i*step];
filters_diff[fbase + 4*step] += temp; //temp_diff[base+i*step]*bottom_data[base+i*step];
}
//			filters_diff[location] += temp_diff[base+i*step]*bottom_data[base+i*step];
//		location = fbase + 3*step;
//		for(int i=0; i<depth; i++)
//			filters_diff[location] += temp_diff[base+i*step]*bottom_data[base+i*step];
//
//		location = fbase + 4*step;
//		for(int i=0; i<depth; i++)
//			filters_diff[location] += temp_diff[base+i*step]*bottom_data[base+i*step];
}*/
if (col + 2 < width)
{
int location = fbase + 2 * step;
for (int i = 0; i < depth; i++)
filters_diff[location] +=
temp_diff[base + i * step] * top_data[base + i * step + 2];
}
/*
else{
int location = fbase + 2*step;
for(int i=0; i<depth; i++)
filters_diff[location] += temp_diff[base+i*step]*bottom_data[base+i*step];
}
*/
}