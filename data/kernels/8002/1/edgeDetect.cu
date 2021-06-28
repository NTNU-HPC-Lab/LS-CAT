#include "includes.h"
__global__ void edgeDetect(unsigned char* device_input_data, unsigned char* device_output_data, int height, int width) {
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;

//check bounds
if (x < 1 || x > width - 1 || y > height - 1 || y < 1)
return;

//for horizontal lines
const int fmat_x[3][3] = {
{-1, 0, 1},
{-2, 0, 2},
{-1, 0, 1}
};
// for vertical lines
const int fmat_y[3][3]  = {
{-1, -2, -1},
{0,   0,  0},
{1,   2,  1}
};

double G_x = 0;
double G_y = 0;
int G;
//go through rows and cols
for (int i = y - 3 / 2; i < y + 3 - 3 / 2; i++) {
for (int j = x - 3 / 2; j < x + 3 - 3 / 2; j++) {
G_x += (double)(fmat_x[i - y + 3 / 2][x - j + 3 / 2] * device_input_data[i * width + j]);
G_y += (double)(fmat_y[i - y + 3 / 2][x - j + 3 / 2] * device_input_data[i * width + j]);
}
}

G = sqrt(G_x * G_x + G_y * G_y);

if (G < 0)
G = 0;
if (G > 255)
G = 255;

device_output_data[y * width + x] = G;
}