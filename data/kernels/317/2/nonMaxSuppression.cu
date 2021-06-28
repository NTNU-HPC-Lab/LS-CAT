#include "includes.h"

#define UPPERTHRESHOLD 90
#define LOWERTHRESHOLD 30

const float G_x[3 * 3] = {
-1, 0, 1,
-2, 0, 2,
-1, 0, 1
};

const float G_y[3 * 3] = {
1, 2, 1,
0, 0, 0,
-1, -2, -1
};

const float gaussian[5 * 5] = {
2.f/159, 4.f/159, 5.f/159, 4.f/159, 2.f/159,
4.f/159, 9.f/159, 12.f/159, 9.f/159, 4.f/159,
5.f/159, 12.f/159, 15.f/159, 12.f/159, 2.f/159,
4.f/159, 9.f/159, 12.f/159, 9.f/159, 4.f/159,
2.f/159, 4.f/159, 5.f/159, 4.f/159, 2.f/159
};








__global__ void nonMaxSuppression(int N, int width, int height, unsigned char * in, unsigned char * out) {
int D = 1;
int x = (blockIdx.x * blockDim.x) + threadIdx.x;
int y = (blockIdx.y * blockDim.y) + threadIdx.y;
if (x >= width || y >= height) {
return;
}
int angle = in[y * width + x];
switch(angle) {
case 0:
if (out[y * width + x] < out[(y + D) * width + x] || out[y * width + x] < out[(y - D) * width + x]) {
out[y * width + x] = 0;
}
break;
case 45:
if (out[y * width + x] < out[(y + D) * width + x - D] || out[y * width + x] < out[(y - D) * width + x + D]) {
out[y * width + x] = 0;
}
break;
case 90:
if (out[y * width + x] < out[y * width + x + D] || out[y * width + x] < out[y * width + x - D]) {
out[y * width + x] = 0;
}
break;

case 135:
if (out[y * width + x] < out[(y + D) * width + x + D] || out[y * width + x] < out[(y - D) * width + x - D]) {
out[y * width + x] = 0;
}
break;
default:
break;
}
}