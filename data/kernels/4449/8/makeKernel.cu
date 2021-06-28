#include "includes.h"
__global__ void makeKernel(float* KernelPhase, int row, int column, float* ImgProperties, float MagXscaling) {
const int numThreads = blockDim.x * gridDim.x;
const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
float MagX = ImgProperties[1];
float pixSize= ImgProperties[0];
float nm = ImgProperties[2];
float lambda = ImgProperties[3];


float pixdxInv = MagX/pixSize*MagXscaling; // Magnification/pixSize
float km = nm/lambda; // nm / lambda

for (int i = threadID; i < row*column; i += numThreads) {
int dx = i%row;
int dy = i/row;

float kdx = float( dx - row/2)*pixdxInv;
float kdy = float( dy - row/2)*pixdxInv;
float temp = km*km - kdx*kdx - kdy*kdy;
KernelPhase[i]= (temp >= 0) ? (sqrtf(temp)-km) : 0;


//This still needs quadrant swapping so this will not work in the ifft routine as is!



}
}