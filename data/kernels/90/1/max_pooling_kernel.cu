#include "includes.h"
__global__ void max_pooling_kernel(float *feature_map, float *probs, float *target, int feature_map_size, int feature_map_num, int pooling_rate, float *rnd_array, int rnd_num){
__shared__ float shFm[16*MAX_POOLING_RATE][16*MAX_POOLING_RATE];

int imgIdx = blockIdx.y / (feature_map_size / 16 / pooling_rate);
int fmIdx = blockIdx.x / (feature_map_size / 16 / pooling_rate);
int tx = (blockIdx.x % (feature_map_size / pooling_rate / 16)) * 16 + threadIdx.x;
int ty = (blockIdx.y % (feature_map_size / pooling_rate / 16)) * 16 + threadIdx.y;
int subsample_size = feature_map_size / pooling_rate;

int rnd_index = ((blockIdx.y * blockDim.y + threadIdx.y) * (blockIdx.x * blockDim.x)  + threadIdx.x ) % rnd_num;
float rnd = rnd_array[rnd_index];

float *fm = feature_map + imgIdx * feature_map_num * feature_map_size * feature_map_size +
fmIdx * feature_map_size * feature_map_size;

probs = probs + imgIdx * feature_map_num * feature_map_size * feature_map_size +
fmIdx * feature_map_size * feature_map_size;

target = target + imgIdx * feature_map_num * subsample_size * subsample_size +
fmIdx * subsample_size * subsample_size;

for(int i = 0; i < pooling_rate; i++){
for(int j = 0; j < pooling_rate; j++){
shFm[threadIdx.y*pooling_rate+i][threadIdx.x*pooling_rate+j] =
fm[(ty*pooling_rate+i) * feature_map_size + (tx*pooling_rate+j)];
}
}

__syncthreads();

float sum = 0;
for(int i = 0; i < pooling_rate; i++){
for(int j = 0; j < pooling_rate; j++){
if(shFm[threadIdx.y*pooling_rate+i][threadIdx.x*pooling_rate+j] > 50){
shFm[threadIdx.y*pooling_rate+i][threadIdx.x*pooling_rate+j] = 50.0f;
}
shFm[threadIdx.y*pooling_rate+i][threadIdx.x*pooling_rate+j] =
__expf(shFm[threadIdx.y*pooling_rate+i][threadIdx.x*pooling_rate+j]);
sum += shFm[threadIdx.y*pooling_rate+i][threadIdx.x*pooling_rate+j];
}
}
for(int i = 0; i < pooling_rate; i++){
for(int j = 0; j < pooling_rate; j++){
shFm[threadIdx.y*pooling_rate+i][threadIdx.x*pooling_rate+j] =
__fdividef(shFm[threadIdx.y*pooling_rate+i][threadIdx.x*pooling_rate+j], (1.0f + sum));
probs[(ty*pooling_rate+i) * feature_map_size + (tx*pooling_rate+j)] =
shFm[threadIdx.y*pooling_rate+i][threadIdx.x*pooling_rate+j];
fm[(ty*pooling_rate+i) * feature_map_size + (tx*pooling_rate+j)] = 0;
}
}

sum = 0;
bool isStop = false;
for(int i = 0; i < pooling_rate && !isStop; i++){
for(int j = 0; j < pooling_rate && !isStop; j++){
sum += shFm[threadIdx.y*pooling_rate+i][threadIdx.x*pooling_rate+j];
if(rnd < sum){
fm[(ty*pooling_rate+i) * feature_map_size + (tx*pooling_rate+j)] = 1;
isStop = true;
}
}
}
if(isStop){
target[threadIdx.y*subsample_size+threadIdx.x] = 1;
}else{
target[threadIdx.y*subsample_size+threadIdx.x] = 0;
}
}