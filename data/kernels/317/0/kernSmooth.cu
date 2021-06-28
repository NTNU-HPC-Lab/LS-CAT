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








__global__ void kernSmooth(int N, int width, int height, unsigned char * in, unsigned char * out, const float * kernel, int kernSize) {
int x = (blockIdx.x * blockDim.x) + threadIdx.x;
int y = (blockIdx.y * blockDim.y) + threadIdx.y;
if (x >= width || y >= height) {
return;
}
float c = 0.0f;
for (int i = 0; i < kernSize; i++) {
int tx = x + i - kernSize/2;
for (int j = 0; j < kernSize; j++) {
int ty = y + j - kernSize/2;
if (tx >= 0 && ty >= 0 && tx < width && ty < height) {
c += in[ty * width + tx] * kernel[j * kernSize + i];
}
}
}
out[y * width + x] = fabs(c);
}