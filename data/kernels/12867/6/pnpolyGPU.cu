#include "includes.h"
__global__ void pnpolyGPU(const float *vertex, float testx, float testy, int* results)
{
int id = blockIdx.x;
int indexOriginX = (blockIdx.x + 1) * 3;
int indexOriginY = (blockIdx.x + 1) * 3 + 1;
int indexDestinoX = blockIdx.x * 3;
int indexDestinoY = blockIdx.x * 3 + 1;

if ( ((vertex[indexOriginY]>testy) != (vertex[indexDestinoY]>testy)) && (testx < (vertex[indexDestinoX]-vertex[indexOriginX]) * (testy-vertex[indexOriginY]) / (vertex[indexDestinoY]-vertex[indexOriginY]) + vertex[indexOriginX]) )
results[id] = 1;
else
results[id] = 0;
}