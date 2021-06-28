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








__global__ void hysteresis(int N, int width, int height, unsigned char * in) {
int x = (blockIdx.x * blockDim.x) + threadIdx.x;
int y = (blockIdx.y * blockDim.y) + threadIdx.y;
if (x >= width || y >= height) {
return;
}
int idx = y * width + x;
if (in[idx] > UPPERTHRESHOLD) {
in[idx] = 255;
} else if (in[idx] < LOWERTHRESHOLD) {
in[idx] = 0;
} else {
for (int dy = -1; dy <= 1; dy++) {
for (int dx = -1; dx <= 1; dx++) {
int nidx = (y + dy) * width + (x + dx);
if(0 <= (y + dy) && (y + dy) < height &&  0 <= (x + dx) && (x + dx) < width && in[nidx] > LOWERTHRESHOLD) {
in[nidx] = 255;
}
}
}
}
}