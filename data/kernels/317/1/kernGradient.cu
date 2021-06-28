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








__global__ void kernGradient(int N, int width, int height, unsigned char * in, unsigned char * gradient, unsigned char * edgeDir, float * G_x, float * G_y) {
int x = (blockIdx.x * blockDim.x) + threadIdx.x;
int y = (blockIdx.y * blockDim.y) + threadIdx.y;
if (x >= width || y >= height) {
return;
}
int idx, dx, dy, tx, ty;
float Gx, Gy, grad, angle;
idx = y * width + x;
Gx = Gy = 0;
for (dy = 0; dy < 3; dy++) {
ty = y + dy - 1;
for (dx = 0; dx < 3; dx++) {
tx = x + dx - 1;
if (tx >= 0 && ty >= 0 && tx < width && ty < height) {
Gx += in[ty * width + tx] * G_x[dy * 3 + dx];
Gy += in[ty * width + tx] * G_y[dy * 3 + dx];
}
}
}
grad = sqrt(Gx * Gx + Gy * Gy);
angle = (atan2(Gx, Gy) / 3.14159f) * 180.0f;
unsigned char roundedAngle;
if (((-22.5 < angle) && (angle <= 22.5)) || ((157.5 < angle) && (angle <= -157.5))) {
roundedAngle = 0;
}
if (((-157.5 < angle) && (angle <= -112.5)) || ((22.5 < angle) && (angle <= 67.5))) {
roundedAngle = 45;
}
if (((-112.5 < angle) && (angle <= -67.5)) || ((67.5 < angle) && (angle <= 112.5))) {
roundedAngle = 90;
}
if (((-67.5 < angle) && (angle <= -22.5)) || ((112.5 < angle) && (angle <= 157.5))) {
roundedAngle = 135;
}
gradient[idx] = grad;
edgeDir[idx] = roundedAngle;
}