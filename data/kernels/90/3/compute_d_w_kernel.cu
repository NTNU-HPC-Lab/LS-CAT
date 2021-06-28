#include "includes.h"
__global__ void compute_d_w_kernel(float *v, float *h, float *dw, bool is_init, int input_size, int lu_padding, int channel_num, int filter_num, int filter_size, int feature_map_size){

int imgIdx = blockIdx.y / (feature_map_size / 32);
int filterIdx = blockIdx.x / (channel_num * feature_map_size / 32);
int channelIdx = (blockIdx.x % (channel_num * feature_map_size / 32)) /
(feature_map_size / 32);
int tx = (blockIdx.x % (channel_num * feature_map_size / 32)) %
(feature_map_size / 32) *32 + threadIdx.x;
int ty = (blockIdx.y % (feature_map_size / 32)) * 32 + threadIdx.y;

__shared__ float shV[32+MAX_FILETER_SIZE][32+MAX_FILETER_SIZE];
__shared__ float shH[32][32];

float sign;
if(is_init){
sign = 1.0f;
}else{
sign = -1.0f;
}

v = v + imgIdx * channel_num * input_size * input_size +
channelIdx * input_size * input_size;

h = h + imgIdx * filter_num * feature_map_size * feature_map_size +
filterIdx * feature_map_size * feature_map_size;

dw = dw + filterIdx * channel_num * filter_size * filter_size +
channelIdx * filter_size * filter_size;

float local_dw = 0.0f;

for(int loadX = 0; loadX <= 32; loadX += filter_size){
for(int loadY = 0; loadY <= 32; loadY += filter_size){
if(loadX < 32 && loadY < 32){
//TODO:feature map overflow
shH[threadIdx.y+loadY][threadIdx.x+loadX] =
h[(ty+loadY)*feature_map_size + (tx+loadX)];
}
if((tx+loadX) < lu_padding ||
(ty+loadY) < lu_padding ||
(tx+loadX) >= (input_size+lu_padding) ||
(ty+loadY) >= (input_size+lu_padding)){
shV[threadIdx.y+loadY][threadIdx.x+loadX] = 0;
}else{
shV[threadIdx.y+loadY][threadIdx.x+loadX] =
v[(ty+loadY-lu_padding)*input_size + (tx+loadX-lu_padding)];
}
}
}

__syncthreads();

for(int i = 0; i < 32; i++){
for(int j = 0; j < 32; j++){
local_dw += shV[threadIdx.y+i][threadIdx.x+j] *
shH[i][j];
}
}

atomicAdd(dw + threadIdx.y*filter_size + threadIdx.x, sign * local_dw);
}