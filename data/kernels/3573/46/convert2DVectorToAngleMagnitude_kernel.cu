#include "includes.h"
__global__ void convert2DVectorToAngleMagnitude_kernel( uchar4 *d_angle_image, uchar4 *d_magnitude_image, float *d_vector_X, float *d_vector_Y, int width, int height, float lower_ang, float upper_ang, float lower_mag, float upper_mag) {
const int x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
const int y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
uchar4 temp_angle, temp_magnitude;

if (x < width && y < height) {
float vector_X = d_vector_X[__mul24(y, width) + x];
float vector_Y = d_vector_Y[__mul24(y, width) + x];

// compute angle and magnitude
float angle = atan2f(vector_Y, vector_X);
float magnitude = vector_X * vector_X + vector_Y * vector_Y;
magnitude = sqrtf(magnitude);

// first draw unmatched pixels in white
if (!isfinite(magnitude)) {
temp_angle.x = 255;
temp_angle.y = 255;
temp_angle.z = 255;
temp_angle.w = 255;
temp_magnitude.x = 255;
temp_magnitude.y = 255;
temp_magnitude.z = 255;
temp_magnitude.w = 255;
} else {
// rescale angle and magnitude from [lower,upper] to [0,1] and convert to
// RGBA jet colorspace

angle -= lower_ang;
angle /= (upper_ang - lower_ang);

float r = 1.0f;
float g = 1.0f;
float b = 1.0f;

if (angle < 0.25f) {
r = 0;
g = 4.0f * angle;
} else if (angle < 0.5f) {
r = 0;
b = 1.0 + 4.0f * (0.25f - angle);
} else if (angle < 0.75f) {
r = 4.0f * (angle - 0.5f);
b = 0;
} else {
g = 1.0f + 4.0f * (0.75f - angle);
b = 0;
}

temp_angle.x = 255.0 * r;
temp_angle.y = 255.0 * g;
temp_angle.z = 255.0 * b;
temp_angle.w = 255;

magnitude -= lower_mag;
magnitude /= (upper_mag - lower_mag);

r = 1.0f;
g = 1.0f;
b = 1.0f;

if (magnitude < 0.25f) {
r = 0;
g = 4.0f * magnitude;
} else if (magnitude < 0.5f) {
r = 0;
b = 1.0 + 4.0f * (0.25f - magnitude);
} else if (magnitude < 0.75f) {
r = 4.0f * (magnitude - 0.5f);
b = 0;
} else {
g = 1.0f + 4.0f * (0.75f - magnitude);
b = 0;
}

temp_magnitude.x = 255.0 * r;
temp_magnitude.y = 255.0 * g;
temp_magnitude.z = 255.0 * b;
temp_magnitude.w = 255;
}

d_angle_image[__mul24(y, width) + x] = temp_angle;
d_magnitude_image[__mul24(y, width) + x] = temp_magnitude;
}
}