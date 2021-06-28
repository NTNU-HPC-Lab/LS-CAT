#include "includes.h"
__global__ void CalculateTransSample( const float *input, float *output, const int wtss, const int htss, const int wts, const int hts, const int ratio ){
const int yts = blockIdx.y * blockDim.y + threadIdx.y;
const int xts = blockIdx.x * blockDim.x + threadIdx.x;
const int curst = wts * yts + xts;

const int yt = yts * ratio, xt = xts * ratio;

if (yts < hts && xts < wts){
for (int i=0; i<ratio; i++){
for (int j=0; j<ratio; j++){
if (yt + i < htss && xt + j < wtss){
const int curt = wtss * (yt+i) + (xt+j);
output[curt*3+0] = input[curst*3+0];
output[curt*3+1] = input[curst*3+1];
output[curt*3+2] = input[curst*3+2];
}
}
}
}
}