#include "includes.h"
__global__ void CalculateSampleT( const float *target, const float *mask, float *subT, int *subM, const int wt, const int ht, const int ws, const int hs, const int sRate ){
const int ys = blockIdx.y * blockDim.y + threadIdx.y;
const int xs = blockIdx.x * blockDim.x + threadIdx.x;
const int curst = ws * ys + xs;

if (ys < hs && xs < ws){
const int yt = ys * sRate;
const int xt = xs * sRate;
int num = 0;
float sum[3] = {0};
int _or = 0;

for (int i=0; i<sRate; i++){
for (int j=0; j<sRate; j++){
if (yt + i < ht && xt + j < wt){
int curt = wt * (yt+i) + (xt+j);
sum[0] += target[curt*3+0];
sum[1] += target[curt*3+1];
sum[2] += target[curt*3+2];
_or |= (mask[curt] > 127.0f);
num++;
}
}
}

subM[curst] = _or;
subT[curst*3+0] += sum[0] / num;
subT[curst*3+1] += sum[1] / num;
subT[curst*3+2] += sum[2] / num;
}
}