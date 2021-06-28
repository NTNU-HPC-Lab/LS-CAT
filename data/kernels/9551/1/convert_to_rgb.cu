#include "includes.h"
/* This code will generate a fractal image. Uses OpenCV, to compile:
nvcc CudaFinal.cu `pkg-config --cflags --libs opencv`  */


typedef enum color {BLUE, GREEN, RED} Color;




__global__ void convert_to_rgb(float *hsv, unsigned char *dest, int width, int heigth, int step, int channels) {
float r, g, b;
float h, s, v;
int ren,col;

ren = blockIdx.x;
col = threadIdx.x;
h = hsv[(ren * step) + (col * channels) + RED];
s = hsv[(ren * step) + (col * channels) + GREEN];
v = hsv[(ren * step) + (col * channels) + BLUE];

float f = h/60.0f;
float hi = floorf(f);
f = f - hi;
float p = v * (1 - s);
float q = v * (1 - s * f);
float t = v * (1 - s * (1 - f));

if(hi == 0.0f || hi == 6.0f) {
r = v;
g = t;
b = p;
} else if(hi == 1.0f) {
r = q;
g = v;
b = p;
} else if(hi == 2.0f) {
r = p;
g = v;
b = t;
} else if(hi == 3.0f) {
r = p;
g = q;
b = v;
} else if(hi == 4.0f) {
r = t;
g = p;
b = v;
} else {
r = v;
g = p;
b = q;
}

dest[(ren * step) + (col * channels) + RED] =  (unsigned char) __float2uint_rn(255.0f * r);
dest[(ren * step) + (col * channels) + GREEN] = (unsigned char) __float2uint_rn(255.0f * g);
dest[(ren * step) + (col * channels) + BLUE] = (unsigned char) __float2uint_rn(255.0f * b);
}