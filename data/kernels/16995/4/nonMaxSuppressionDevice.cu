#include "includes.h"

# define MAX(a, b) ((a) > (b) ? (a) : (b))

# define GAUSSIAN_KERNEL_SIZE 3
# define SOBEL_KERNEL_SIZE 5
# define TILE_WIDTH 32
# define SMEM_SIZE 128
__global__ void nonMaxSuppressionDevice(int width, int height, float *d_gradientX, float *d_gradientY, float* d_gradientMag, float* d_nonMax) {
int ix = threadIdx.x + blockIdx.x * blockDim.x;
int iy = threadIdx.y + blockIdx.y * blockDim.y;

if (ix < width && iy < height) {
int tid = iy * width + ix;

float d_gradientMag_tid = d_gradientMag[tid];
float d_gradientMag_tid_next = d_gradientMag[tid + 1];
float d_gradientMag_tid_prev = d_gradientMag[tid - 1];
float d_gradientMag_tid_width_next = d_gradientMag[tid + width + 1];
float d_gradientMag_tid_width_prev = d_gradientMag[tid - width - 1];
float d_gradientMag_tid_width_plus = d_gradientMag[tid + width];
float d_gradientMag_tid_width_minus = d_gradientMag[tid - width];
float d_gradientMag_tid_width_minus_next = d_gradientMag[tid - width + 1];
float d_gradientMag_tid_width_plus_prev = d_gradientMag[tid + width - 1];

float d_gradientXT = d_gradientX[tid];
float d_gradientYT = d_gradientY[tid];

float tanYX;
float magB, magA;

if ((tid < width) || (tid >= ((height - 1) * width))) // Top and Bottom Edge
d_nonMax[tid] = 0;
else if ((tid % width == 0) || (tid % width == (width - 1))) // Left and Right Edge
d_nonMax[tid] = 0;
else {
if (d_gradientMag_tid == 0)
d_nonMax[tid] = 0;
else if (d_gradientXT >= 0) { // Direction East
if (d_gradientYT >= 0) { // Direction South-East
if (d_gradientXT >= d_gradientYT) { // East of South-East direction
tanYX = (float)(d_gradientYT / d_gradientXT);

magA = ((1 - tanYX) * d_gradientMag_tid_next) + (tanYX * d_gradientMag_tid_width_next);
magB = ((1 - tanYX) * d_gradientMag_tid_prev) + (tanYX * d_gradientMag_tid_width_plus_prev);
}
else { // South of South-East direction
tanYX = (float)(d_gradientXT / d_gradientYT);

magA = ((1 - tanYX) * d_gradientMag_tid_width_plus) + (tanYX * d_gradientMag_tid_width_next);
magB = ((1 - tanYX) * d_gradientMag_tid_width_minus) + (tanYX * d_gradientMag_tid_width_prev);
}
}
else { // Direction North-East
if (d_gradientXT >= (-1 * d_gradientYT)) { // East of North-East direction
tanYX = (float)((-1 * d_gradientYT) / d_gradientXT);

magA = ((1 - tanYX) * d_gradientMag_tid_next) + (tanYX * d_gradientMag_tid_width_minus_next);
magB = ((1 - tanYX) * d_gradientMag_tid_prev) + (tanYX * d_gradientMag_tid_width_plus_prev);
}
else { // North of North-East direction
tanYX = (float)(d_gradientXT / (-1 * d_gradientYT));

magA = ((1 - tanYX) * d_gradientMag_tid_width_plus) + (tanYX * d_gradientMag_tid_width_plus_prev);
magB = ((1 - tanYX) * d_gradientMag_tid_width_minus) + (tanYX * d_gradientMag_tid_width_minus_next);
}
}
}
else { // Direction West
if (d_gradientYT >= 0) { // Direction South-West
if (d_gradientYT >= (-1 * d_gradientXT)) { // South of South-West direction
tanYX = (float)((-1 * d_gradientXT) / d_gradientYT);
magA = ((1 - tanYX) * d_gradientMag_tid_width_plus) + (tanYX * d_gradientMag_tid_width_plus_prev);
magB = ((1 - tanYX) * d_gradientMag_tid_width_minus) + (tanYX * d_gradientMag_tid_width_minus_next);
}
else { // West of South-West direction
tanYX = (float)(d_gradientYT / (-1 * d_gradientXT));
magA = ((1 - tanYX) * d_gradientMag_tid_prev) + (tanYX * d_gradientMag_tid_width_plus_prev);
magB = ((1 - tanYX) * d_gradientMag_tid_next) + (tanYX * d_gradientMag_tid_width_minus_next);
}
}
else { // Direction North-West
if (d_gradientYT >= d_gradientXT) { // West of North-West direction
tanYX = (float)(d_gradientYT / d_gradientXT);
magA = ((1 - tanYX) * d_gradientMag_tid_prev) + (tanYX * d_gradientMag_tid_width_prev);
magB = ((1 - tanYX) * d_gradientMag_tid_next) + (tanYX * d_gradientMag_tid_width_next);
}
else {// North of North-West direction
tanYX = (float)(d_gradientXT / d_gradientYT);
magA = ((1 - tanYX) * d_gradientMag_tid_width_plus) + (tanYX * d_gradientMag_tid_width_next);
magB = ((1 - tanYX) * d_gradientMag_tid_width_minus) + (tanYX * d_gradientMag_tid_width_prev);
}
}
}

if ((d_gradientMag_tid < magA) || (d_gradientMag_tid < magB))
d_nonMax[tid] = 0;
else
d_nonMax[tid] = d_gradientMag_tid;
}
}
}