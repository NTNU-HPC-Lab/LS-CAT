#include "includes.h"
__global__ void kern_BlurBuffer(float* input, float* output, int size, int spread, int dim)
{
int idx = CUDASTDOFFSET;
int x = (idx / spread) % dim;
float curr = input[idx];
float down = (idx-spread >= 0)   ? input[idx-spread] : 0;
float up   = (idx+spread < size) ? input[idx+spread] : 0;
float newVal = 0.7865707f * curr + 0.1064508f * ((x > 0 ? down : curr) + (x < dim-1 ? up : curr));
__syncthreads();
if( idx < size )
{
output[idx] = newVal;
}
}