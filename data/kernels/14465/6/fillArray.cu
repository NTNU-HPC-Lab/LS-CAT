#include "includes.h"



#define A 1.2f
#define B 0.5f
#define MIN_LEARNING_RATE 0.000001f
#define MAX_LEARNING_RATE 50.0f

// Device functions

// Array[height * width]
__global__ void fillArray(float *array, float value, int arrayLength)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i >= arrayLength)
return;

array[i] = value;
}