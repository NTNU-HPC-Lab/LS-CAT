#include "includes.h"
__global__ void calibrate_fix2float(float * dst, const float* sA, const float* sB, float alpha, float beta, int height, int width, int threads) {
int ri = blockIdx.x;
int tid = threadIdx.x;
int loop = (width / threads) + ((width % threads == 0) ? 0 : 1);

float rscale = (sA[ri] == 0.0f) ? 1.0f : sA[ri];
float * data = dst + width * ri;
int idx = 0;
for (int i = 0; i < loop; ++i) {
if(idx + tid < width){
float temp = data[idx + tid];
float cscale = (sB[idx + tid] == 0.0f) ? 255.0f : sB[idx + tid];
data[idx + tid] = beta  * temp + alpha * temp * rscale * cscale;
}
idx += threads;
}
}