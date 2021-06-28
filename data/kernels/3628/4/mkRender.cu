#include "includes.h"
__global__ void mkRender(float *fb, int max_x, int max_y) {
//MK: Pixel 위치 계산을 위해 ThreadId, BlockId를 사용함
int i = threadIdx.x + blockIdx.x * blockDim.x;
int j = threadIdx.y + blockIdx.y * blockDim.y;

//MK: 계산된 Pixel 위치가 FB사이즈 보다 크면 연산을 수행하지 않음
if((i >= max_x) || (j >= max_y)){
return;
}

//MK: FB Pixel 값 계산
int pixel_index = j*max_x*3 + i*3;
fb[pixel_index + 0] = float(i) / max_x;
fb[pixel_index + 1] = float(j) / max_y;
fb[pixel_index + 2] = 0.2f;
}