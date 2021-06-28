#include "includes.h"



struct cudaGraphicsResource* cuda_vbo_resource;


__global__ void mandelbrotLogicKernel(float* data, int width, int height, const int maxIteration, const double middlea, const double middleb, const double rangea, const double rangeb)
{
unsigned int x_dim = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int y_dim = blockIdx.y * blockDim.y + threadIdx.y;

int ipos = width * y_dim * 3 + x_dim * 3;

double x0 = (double)x_dim / (double)width;
double y0 = (double)y_dim / (double)height;

x0 = x0 * rangea + middlea - rangea / 2;
y0 = y0 * rangeb + middleb - rangeb / 2;

double real = 0;
double imaginary = 0;

int iteration = 0;
while (real * real + imaginary * imaginary <= 4 && iteration < maxIteration)
{
double temp = real * real - imaginary * imaginary + x0;
imaginary = 2 * real * imaginary + y0;
real = temp;
iteration++;
}

// Color algorithm from my brother (https://github.com/Julian-Wollersberger/Apfelmannchen)
int runde;
double fraction;

if (iteration == maxIteration)
{
data[ipos + 0] = 1.0f;
data[ipos + 1] = 1.0f;
data[ipos + 2] = 1.0f;
}
else
{
iteration += 8;
runde = 15;
while (iteration >= runde)
runde = (runde * 2) + 1;

fraction = (iteration - runde / 2) / (double)(runde / 2);

if (fraction < 0)
{
data[ipos + 0] = 1.0f;
data[ipos + 1] = 1.0f;
data[ipos + 2] = 1.0f;
}
else if (fraction < 1.0 / 6) { data[ipos + 0] = 1.0f;  data[ipos + 1] = 0.0f;  data[ipos + 2] = fraction * 6.0f; }
else if (fraction < 2.0 / 6) { data[ipos + 0] = 1 - (fraction - 1.0 / 6) * 6;  data[ipos + 1] = 0.0f;  data[ipos + 2] = 1.0f; }
else if (fraction < 3.0 / 6) { data[ipos + 0] = 0.0f;  data[ipos + 1] = (fraction - 2.0 / 6) * 6;  data[ipos + 2] = 1.0f; }
else if (fraction < 4.0 / 6) { data[ipos + 0] = 0.0f;  data[ipos + 1] = 1.0f;  data[ipos + 2] = 1 - (fraction - 3.0 / 6) * 6; }
else if (fraction < 5.0 / 6) { data[ipos + 0] = (fraction - 4.0 / 6) * 6;  data[ipos + 1] = 1.0f;  data[ipos + 2] = 0.0f; }
else if (fraction <= 6.0 / 6) { data[ipos + 0] = 1.0f;  data[ipos + 1] = 1 - (fraction - 5.0 / 6) * 6;  data[ipos + 2] = 0.0f; }
else
{
data[ipos + 0] = 0.0f;
data[ipos + 1] = 0.0f;
data[ipos + 2] = 0.0f;
}
}
}