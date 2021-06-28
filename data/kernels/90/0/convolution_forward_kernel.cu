#include "includes.h"
__global__ void convolution_forward_kernel(float *input, float *filters, float *feature_map, float *hbias, int input_size, int channel_num, int feature_map_size, int filter_size, int filter_num, int lu_padding, float sigma){
__shared__ float shImg[32+MAX_FILETER_SIZE-1][32+MAX_FILETER_SIZE-1];
__shared__ float shFilter[MAX_FILETER_SIZE][MAX_FILETER_SIZE];

int imgIdx = blockIdx.y / (input_size / 32);
int filterIdx = blockIdx.x / (input_size / 32);
int tx = blockIdx.x % (input_size / 32) * 32 + threadIdx.x;
int ty = blockIdx.y % (input_size / 32) * 32 + threadIdx.y;

float *target = feature_map +
imgIdx * feature_map_size * feature_map_size * filter_num +
feature_map_size * feature_map_size * filterIdx +
ty * feature_map_size + tx;

float local_target = 0.0f;

for(int g = 0; g < channel_num; g++){

if(threadIdx.x < filter_size && threadIdx.y < filter_size){
shFilter[threadIdx.y][threadIdx.x] =
filters[filterIdx * channel_num * filter_size * filter_size +
+ g * filter_size * filter_size +
threadIdx.y * filter_size + threadIdx.x];
}
__syncthreads();

float *img = input + imgIdx * input_size * input_size * channel_num
+ g * input_size * input_size;

float *shImgLoad = &shImg[threadIdx.y][threadIdx.x];
if(tx < lu_padding || ty < lu_padding){
*shImgLoad = 0;
}else{
*shImgLoad = img[(ty-lu_padding) * input_size + (tx-lu_padding)];
}

if(threadIdx.x < MAX_FILETER_SIZE-1){
shImgLoad = &shImg[threadIdx.y][threadIdx.x+32];
if(ty < lu_padding || (tx+32) >= (input_size+lu_padding)){
*shImgLoad = 0;
}else{
*shImgLoad = img[(ty-lu_padding) * input_size +
(tx+32-lu_padding)];
}
}

if(threadIdx.y < MAX_FILETER_SIZE-1){
shImgLoad = &shImg[threadIdx.y+32][threadIdx.x];
if(tx < lu_padding || (ty+32) >= (input_size+lu_padding)){
*shImgLoad = 0;
}else{
*shImgLoad = img[(ty+32-lu_padding) * input_size +
(tx-lu_padding)];
}

if(threadIdx.x < MAX_FILETER_SIZE-1){
shImgLoad = &shImg[threadIdx.y+32][threadIdx.x+32];
if((ty+32) >= (input_size+lu_padding) ||
(tx+32) >= (input_size+lu_padding)){
*shImgLoad = 0;
}else{
*shImgLoad = img[(ty+32-lu_padding) * input_size +
(tx+32-lu_padding)];
}
}
}
__syncthreads();

float *imgPtr = &shImg[threadIdx.y][threadIdx.x];

for(int i = 0; i < filter_size; i++){
for(int j = 0; j < filter_size; j++){
local_target += imgPtr[j] * shFilter[i][j];
}
imgPtr += 32 + MAX_FILETER_SIZE - 1;
}

__syncthreads();

}

local_target += hbias[filterIdx];
local_target *= __fdividef(1.0f , sigma * sigma);
*target = local_target;

}