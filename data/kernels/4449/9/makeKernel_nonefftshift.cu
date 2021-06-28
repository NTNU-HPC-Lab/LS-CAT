#include "includes.h"
__global__ void makeKernel_nonefftshift(float* KernelPhase, int row, int column, float* ImgProperties) {
const int numThreads = blockDim.x * gridDim.x;
const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
float pixSize = ImgProperties[0];
float MagX = ImgProperties[1];
float nmed = ImgProperties[2];
float lambda = ImgProperties[3];
float MagXscaling = 1/ImgProperties[4];
float pixdxInv = MagX / pixSize*MagXscaling; // Magnification/pixSize
float km = nmed / lambda; // nmed / lambda


for (int i = threadID; i < row*column; i += numThreads) {
int dx = i % row;
int dy = i / row;

dx= ((dx - row / 2)>0) ? (dx - row) : dx;
dy= ((dy - row / 2)>0) ? (dy - row) : dy;

float kdx = float(dx)*pixdxInv;
float kdy = float(dy)*pixdxInv;
float temp = km*km - kdx*kdx - kdy*kdy;
KernelPhase[i] = (temp >= 0) ? (sqrtf(temp)-km) : 0;
}
}