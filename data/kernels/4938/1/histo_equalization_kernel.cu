#include "includes.h"
__global__ void histo_equalization_kernel ( unsigned char *buffer, long size, int *histo, unsigned char *output ) {

int i = threadIdx.x + blockIdx.x * blockDim.x;
int offset = blockDim.x * gridDim.x;
while (i < size) {
if ( dev_lut[buffer[i]] > 255)
output[i] = 255;
else
output[i] = (unsigned char) dev_lut[buffer[i]];

i += offset;
}
}