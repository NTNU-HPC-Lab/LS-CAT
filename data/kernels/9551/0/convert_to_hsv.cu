#include "includes.h"
/* This code will generate a fractal image. Uses OpenCV, to compile:
nvcc CudaFinal.cu `pkg-config --cflags --libs opencv`  */


typedef enum color {BLUE, GREEN, RED} Color;




__global__ void convert_to_hsv(unsigned char *src, float *hsv, int width, int heigth, int step, int channels) {
float r, g, b;
float h, s, v;
int ren,col;

ren = blockIdx.x;
col = threadIdx.x;

r = src[(ren * step) + (col * channels) + RED] / 255.0f;
g = src[(ren * step) + (col * channels) + GREEN] / 255.0f;
b = src[(ren * step) + (col * channels) + BLUE] / 255.0f;

float max = fmax(r, fmax(g, b));
float min = fmin(r, fmin(g, b));
float diff = max - min;

v = max;

if(v == 0.0f) { // black
h = s = 0.0f;
} else {
s = diff / v;
if(diff < 0.001f) { // grey
h = 0.0f;
} else { // color
if(max == r) {
h = 60.0f * (g - b)/diff;
if(h < 0.0f) { h += 360.0f; }
} else if(max == g) {
h = 60.0f * (2 + (b - r)/diff);
} else {
h = 60.0f * (4 + (r - g)/diff);
}
}
}
// confusion line
float minh=40.0f;
float maxh=200.0f;
float minis = 0;
float maxs = 100;
float miniv = 0;
float maxv = 100;

// if conditionals to check the color blindness line, if the pixel is in this line i change the color to other color base shifting the h
if (h > minh && h < maxh && s > minis && s < maxs && v > miniv && v < maxv){

hsv[(ren * step) + (col * channels) + RED] =  (float) (h + 140.0f);
hsv[(ren * step) + (col * channels) + GREEN] = (float) (s);
hsv[(ren * step) + (col * channels) + BLUE] = (float) (v);
} else { // this keep the pixel if it is out of the color blindnessline
hsv[(ren * step) + (col * channels) + RED] =  (float) (h);
hsv[(ren * step) + (col * channels) + GREEN] = (float) (s);
hsv[(ren * step) + (col * channels) + BLUE] = (float) (v);
}


}