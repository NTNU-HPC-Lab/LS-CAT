#include "includes.h"
__device__ bool checkBoundary(int blockIdx, int blockDim, int threadIdx){
int x = threadIdx;
int y = blockIdx;
return (x == 0 || x == (blockDim-1) || y == 0 || y == 479);
}
__global__ void mAdvect(float *new_data, float *old_data, float *xv, float *yv, float t_step, float s_stepX, float s_stepY) {
if(checkBoundary(blockIdx.x, blockDim.x, threadIdx.x)) return;
int Idx = blockIdx.x * blockDim.x + threadIdx.x;
float curr_x = (float)threadIdx.x;
float curr_y = (float)blockIdx.x;
float last_x = curr_x - t_step*s_stepX*xv[Idx];
float last_y = curr_y - t_step*s_stepY*yv[Idx];

if(last_x < 1.5)   last_x = 1.5;
if(last_x > 637.5) last_x = 637.5;
if(last_y < 1.5)   last_y = 1.5;
if(last_y > 477.5) last_y = 477.5;

// Bilinear Interpolation
float xDiff = last_x - (int)last_x;
float yDiff = last_y - (int)last_y;
int LeftTopX = (int)last_x;
int LeftTopY = (int)last_y;
int LeftTopIdx = LeftTopY * blockDim.x + LeftTopX;
new_data[Idx] = (xDiff*yDiff)*old_data[LeftTopIdx+blockDim.x+1]
+(xDiff*(1.f-yDiff))*old_data[LeftTopIdx+1]
+((1.f-xDiff)*yDiff)*old_data[LeftTopIdx+blockDim.x]
+((1.f-xDiff)*(1.f-yDiff))*old_data[LeftTopIdx];
}