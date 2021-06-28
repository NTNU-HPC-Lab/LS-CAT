#include "includes.h"
__global__ void PossionImageCloningIteration( const float *fixed, const float *mask, float *input, float *output, const int wt, const int ht ){
const int dir[4][2] = {	{0, -1}, {1, 0}, {0, 1}, {-1, 0}};
int num = 4;
const int yt = blockIdx.y * blockDim.y + threadIdx.y;
const int xt = blockIdx.x * blockDim.x + threadIdx.x;
const int curt = wt * yt + xt;
if (yt < ht && xt < wt && mask[curt] > 127.0f){
float sum[3] = {0};
for (int i=0; i<4; i++){
int dxt = xt + dir[i][0];
int dyt = yt + dir[i][1];
int dcurt = wt * dyt + dxt;
if (dxt >= 0 && dxt < wt && dyt >= 0 && dyt < ht &&
mask[dcurt] > 127.0f){
sum[0] += input[dcurt*3+0];
sum[1] += input[dcurt*3+1];
sum[2] += input[dcurt*3+2];
}
}
output[curt*3+0] = fixed[curt*3+0] + sum[0] / num;
output[curt*3+1] = fixed[curt*3+1] + sum[1] / num;
output[curt*3+2] = fixed[curt*3+2] + sum[2] / num;
}
}