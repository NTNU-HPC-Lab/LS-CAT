#include "includes.h"
__global__ void CalculateSampleB( const float *background, float *subBG, const int wb, const int hb, const int ws, const int hs, const int sRate ){
const int ys = blockIdx.y * blockDim.y + threadIdx.y;
const int xs = blockIdx.x * blockDim.x + threadIdx.x;
const int curst = ws * ys + xs;

if (ys < hs && xs < ws){
const int yb = ys * sRate;
const int xb = xs * sRate;
int num = 0;
float sum[3] = {0};

for (int i=0; i<sRate; i++){
for (int j=0; j<sRate; j++){
if (yb + i < hb && xb + j < wb){
int curb = wb * (yb+i) + (xb+j);
sum[0] += background[curb*3+0];
sum[1] += background[curb*3+1];
sum[2] += background[curb*3+2];
num++;
}
}
}
subBG[curst*3+0] = sum[0] / num;
subBG[curst*3+1] = sum[1] / num;
subBG[curst*3+2] = sum[2] / num;
}
}