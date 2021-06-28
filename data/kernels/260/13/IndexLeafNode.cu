#include "includes.h"
__global__ void IndexLeafNode(const char *text, bool *forest, int text_size, int step)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int offset = blockIdx.x*step+blockDim.x;
forest[offset+threadIdx.x] = (text[idx] != '\n' && idx < text_size);
}