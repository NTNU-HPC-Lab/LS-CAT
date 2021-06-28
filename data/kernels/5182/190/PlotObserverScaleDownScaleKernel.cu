#include "includes.h"
__global__ void PlotObserverScaleDownScaleKernel(float* history, int nbCurves, int size)
{
int id = blockDim.x*blockIdx.y*gridDim.x
+ blockDim.x*blockIdx.x
+ threadIdx.x;

if (id >= size)
return;

int baseAddress = 2 * id;
float val1 = history[baseAddress];
float val2 = history[baseAddress + nbCurves];
float average = (val1 + val2) / 2;
history[id] = average;
}