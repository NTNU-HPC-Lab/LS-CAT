#include "includes.h"
__global__ void jacobiFirstLocal(float *x, const float *diagonal_values , const float *non_diagonal_values, const int *indeces ,const float *y, const int size)
{
const int index = blockIdx.x * blockDim.x + threadIdx.x;
float error = 1 ;
float current_value = 1 ;

if (index < size)
{
float local_diagonal_value ;
float local_non_diagonal_values[2];
int local_indeces[2];
float local_y;

local_diagonal_value = diagonal_values[index];
local_non_diagonal_values[0] = non_diagonal_values[2*index];
local_non_diagonal_values[1] = non_diagonal_values[2*index+1];
local_indeces[0] = indeces[2*index];
local_indeces[1] = indeces[2*index+1];
local_y = y[index];

float sum = 0 ;

while(fabsf(error) > 0.00001)
{
for (int i = 0 ; i<2 ; i++)
{
sum += local_non_diagonal_values[i]  * x[local_indeces[i]] ;
}

current_value = (local_y - sum )/local_diagonal_value;
error = current_value - x[index] ;
x[index] = current_value ;
sum = 0 ;
__syncthreads();
}
}
}