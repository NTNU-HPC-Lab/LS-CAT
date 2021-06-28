#include "includes.h"
__global__ void _roll_array( const float* array, const long* step, float* new_array, const int b, const int n, const int d ) {
int index = threadIdx.x + blockIdx.x * blockDim.x;
if (index >= b * n * d)
return;

const int c_b = index / (n * d);
const int c_n = (index - c_b * n * d) / d;
const int c_d = index % d;

const float c_array_element = array[c_b * n * d + c_n * d + c_d];
float* c_new_array = &new_array[c_b * n * d];

int c_step = int(step[c_b]);
int new_n = ((c_n + c_step) % n + n) % n;
int position = new_n * d + c_d;

c_new_array[position] = c_array_element;
}