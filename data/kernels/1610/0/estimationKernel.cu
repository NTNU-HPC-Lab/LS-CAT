#include "includes.h"
/*
There can be problem with crashing app
It is caused by WDDM TDR delay
this delay works in such a way that kill the kernel if it doesnt finish in specific time
so for big numbers it can be a problem
but you can change time or even turn it off in Nsight monitor : option->general->microsoft display driver
*/




#define PI 3.14159265358979323846


#define N	10000	//data size
#define ES	10000	//estimation size
#define HS	20		//histogram size	the lower hs is the better results will appear
//do not spoil and dont set data size greater than histogram size


__global__ void estimationKernel(float* data, size_t n, float* kernelEstimation, size_t es, float dx, float h)
{
int index = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;

for (int i = index; i < es; i += stride)
{
float di;
di = dx * i;	//on which position on OX axis we calculate the estimation

kernelEstimation[i] = 0;
for (int j = 0; j < n; j++)
{
//formula:
float power = -0.5f * (di - data[j]) * (di - data[j]) / h / h;
kernelEstimation[i] += expf(power);

}
kernelEstimation[i] /= (n * h) * sqrt(2 * PI);	//also formula

}
}