#include "includes.h"


#define ARRAY_SIZE 200
#define ARRAY_BYTES ARRAY_SIZE * sizeof(float)


__global__ void CalculateSquare(float* p_out, float* p_in)
{
int index = threadIdx.x;
float valueToSuqare = p_in[index];
p_out[index] = valueToSuqare * valueToSuqare;
}