#include "includes.h"
__global__ void FindMinMax(float *d_MinMax, float *d_Data, int width, int height)
{
__shared__ float minvals[128];
__shared__ float maxvals[128];
const int tx = threadIdx.x;
const int x = __mul24(blockIdx.x, 128) + tx;
const int y = __mul24(blockIdx.y, 16);
const int b = blockDim.x;
int p = __mul24(y, width) + x;
if (x<width) {
float val = d_Data[p];
minvals[tx] = val;
maxvals[tx] = val;
} else {
float val = d_Data[p-x];
minvals[tx] = val;
maxvals[tx] = val;
}
for (int ty=1;ty<16;ty++) {
p += width;
if (tx<width) {
float val = d_Data[p];
if (val<minvals[tx])
minvals[tx] = val;
if (val>maxvals[tx])
maxvals[tx] = val;
}
}
__syncthreads();
int mod = 1;
for (int d=1;d<b;d<<=1) {
if ((tx&mod)==0) {
if (minvals[tx+d]<minvals[tx+0])
minvals[tx+0] = minvals[tx+d];
if (maxvals[tx+d]>maxvals[tx+0])
maxvals[tx+0] = maxvals[tx+d];
}
mod = 2*mod + 1;
__syncthreads();
}
if (tx==0) {
int ptr = 2*(__mul24(gridDim.x,blockIdx.y) + blockIdx.x);
d_MinMax[ptr+0] = minvals[0];
d_MinMax[ptr+1] = maxvals[0];
}
__syncthreads();
}