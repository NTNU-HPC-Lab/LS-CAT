#include "includes.h"
using namespace std;
#define eps 1e-4

//每个thread负责output的一个pixel

__global__ void convolution2d(float *img, float *kernel, float* result, int n, int m, int kw, int kh, int out_n, int out_m, bool padding)
{
int bx = blockIdx.x, by = blockIdx.y;
int tx = threadIdx.x, ty = threadIdx.y;
int x = bx * blockDim.x + tx;
int y = by * blockDim.y + ty;
int idx = y * out_m + x;
//printf("%d %d %d %d %d %d\n", bx, by, tx, ty, x, y);
if(idx < out_n * out_m){
float ret = 0;
for(int i = 0; i < kw; i++){
for(int j = 0; j < kh; j++){
//ret += img[(y + j) * m + (x + i)] * kernel[i * kh + j];
//padding = same: (x,y) 为中心点，(x-kw/2, y-kh/2)为左上角第一个点
//padding = valid: (x+kw/2, y+kh/2)为中心点, (x,y)为左上角第一个点
int cur_x = 0, cur_y = 0;
if(padding == true){
cur_x = x - kw / 2 + i;
cur_y = y - kh / 2 + j;
}
else{
cur_x = x + i;
cur_y = y + j;
}
if(cur_x >= 0 and cur_x < n and cur_y >= 0 and cur_y < m){
ret += img[cur_y * m + cur_x] * kernel[i * kh + j];
}
}
}
//printf("%d %d %d %f\n", x, y, idx, ret);
//__syncthreads();
result[idx] = ret;
}
}