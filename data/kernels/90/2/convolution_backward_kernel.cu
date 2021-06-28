#include "includes.h"
__global__ void convolution_backward_kernel(float *y_h, float *filters, float *vbias, float *target, float *y_v, int input_size, int lu_padding, int channel_num, int feature_map_size, int filter_num, int filter_size, float *rnd_array, int rnd_num){
int imgIdx = blockIdx.y / (input_size / 16);
int channelIdx = blockIdx.x / (input_size / 16);
int tx = (blockIdx.x % (input_size / 16)) * 16 + threadIdx.x;
int ty = (blockIdx.y % (input_size / 16)) * 16 + threadIdx.y;
int padding = (filter_size - 1);

int rnd_index = ((blockIdx.y * blockDim.y + threadIdx.y) * (blockIdx.x * blockDim.x)  + threadIdx.x ) % rnd_num;
float rnd = rnd_array[rnd_index];

__shared__ float shHidden[16+2*(MAX_FILETER_SIZE-1)][16+2*(MAX_FILETER_SIZE-1)];
__shared__ float shFlipFilter[MAX_FILETER_SIZE][MAX_FILETER_SIZE];
float local_target = 0.0f;

target = target + imgIdx * channel_num * input_size * input_size +
channelIdx * input_size * input_size;

float *target_y_v = y_v + imgIdx * channel_num * input_size * input_size +
channelIdx * input_size * input_size;

__syncthreads();


for(int f = 0; f < filter_num; f++){
float *cur_y_h = y_h + imgIdx * filter_num * feature_map_size * feature_map_size +
f * feature_map_size * feature_map_size;

float *cur_filter = filters + f * channel_num * filter_size * filter_size +
channelIdx * filter_size * filter_size;

if(threadIdx.x < filter_size && threadIdx.y < filter_size){
shFlipFilter[threadIdx.y][threadIdx.x] =
cur_filter[(filter_size-1-threadIdx.y)*filter_size + filter_size-1-threadIdx.x];
}

float *shHiddenLoad = &shHidden[threadIdx.y][threadIdx.x];
if(tx < padding || ty < padding){
*shHiddenLoad = 0;
}else{
*shHiddenLoad = cur_y_h[(ty-padding) * input_size +
(tx-padding)];
}

if(threadIdx.x < 2 * padding){
shHiddenLoad = &shHidden[threadIdx.y][threadIdx.x+16];
if(ty < padding || (tx+16) >= (feature_map_size+padding)){
*shHiddenLoad = 0;
}else{
*shHiddenLoad = cur_y_h[(ty-padding) * feature_map_size +
(tx+16-padding)];
}
}

if(threadIdx.y < 2 * padding){
shHiddenLoad = &shHidden[threadIdx.y+16][threadIdx.x];
if(tx < padding || (ty+16) >= (feature_map_size+padding)){
*shHiddenLoad = 0;
}else{
*shHiddenLoad = cur_y_h[(ty+16-padding) * feature_map_size +
(tx-padding)];
}

if(threadIdx.x < 2 * padding){
shHiddenLoad = &shHidden[threadIdx.y+16][threadIdx.x+16];
if((ty+16) >= (feature_map_size+padding) ||
(tx+16) >= (feature_map_size+padding)){
*shHiddenLoad = 0;
}else{
*shHiddenLoad = cur_y_h[(ty+16-padding) * feature_map_size +
(tx+16-padding)];
}
}
}

__syncthreads();

for(int i = 0; i < filter_size; i++){
for(int j = 0; j < filter_size; j++){
target[ty*input_size+tx] +=
local_target +=
shHidden[threadIdx.y+i+lu_padding][threadIdx.x+j+lu_padding] *
shFlipFilter[i][j];
}
}

__syncthreads();
}
local_target += vbias[channelIdx];
//local_target = expf(-local_target);
//local_target = __fdividef(1.0f , (1.0f + local_target));
if(rnd < local_target){
target_y_v[ty*input_size+tx] = 1;
}else{
target_y_v[ty*input_size+tx] = 0;
}
target[ty*input_size+tx] = local_target;
}