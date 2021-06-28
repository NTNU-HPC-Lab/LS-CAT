#include "includes.h"
__global__ void rgbUtoLab3F_kernel(int width, int height, float gamma, unsigned int* rgbU, float* devL, float* devA, float* devB) {
int x0 = blockDim.x * blockIdx.x + threadIdx.x;
int y0 = blockDim.y * blockIdx.y + threadIdx.y;
if ((x0 < width) && (y0 < height)) {
int index = y0 * width + x0;
unsigned int rgb = rgbU[index];
float r = (float)(rgb & 0xff)/255.0;
float g = (float)((rgb & 0xff00) >> 8)/255.0;
float b = (float)((rgb & 0xff0000) >> 16)/255.0;
r = powf(r, gamma);
g = powf(g, gamma);
b = powf(b, gamma);
float x = (0.412453 * r) +  (0.357580 * g) + (0.180423 * b);
float y = (0.212671 * r) +  (0.715160 * g) + (0.072169 * b);
float z = (0.019334 * r) +  (0.119193 * g) + (0.950227 * b);
/*D65 white point reference */
const float x_ref = 0.950456;
const float y_ref = 1.000000;
const float z_ref = 1.088754;
/* threshold value  */
const float threshold = 0.008856;
x = x / x_ref;
y = y / y_ref;
z = z / z_ref;

float fx =
(x > threshold) ? powf(x,(1.0/3.0)) : (7.787*x + (16.0/116.0));
float fy =
(y > threshold) ? powf(y,(1.0/3.0)) : (7.787*y + (16.0/116.0));
float fz =
(z > threshold) ? powf(z,(1.0/3.0)) : (7.787*z + (16.0/116.0));
/* compute Lab color value */
devL[index] =
(y > threshold) ? (116*powf(y,(1.0/3.0)) - 16) : (903.3*y);
devA[index] = 500.0f * (fx - fy);
devB[index] = 200.0f * (fy - fz);
}
}