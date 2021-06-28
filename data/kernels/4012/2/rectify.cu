#include "includes.h"
__global__ void rectify(unsigned char* image, unsigned height, unsigned width, int thread_count)
{
// process image
int block = (height * width * 4) / thread_count;
int offset = threadIdx.x * block;
for (int i = 0; i < block; i++)
{
int j = offset + i;
if (image[j] < 127)	image[j] = 127;

}

}