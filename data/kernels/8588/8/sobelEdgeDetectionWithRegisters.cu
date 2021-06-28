#include "includes.h"
__global__ void sobelEdgeDetectionWithRegisters (int *input, int *output, int width, int height, int thresh) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;
int index = j * width + i;

int val1 = input[width * (j - 1) + (i + 1)], val2 = input[width * (j - 1) + (i - 1)], val3 = input[width * (j + 1) + (i + 1)], val4 = input[width * (j + 1) + (i - 1)];

if ( ((i > 0) && (j > 0)) && ((i < (width - 1)) && (j < (height - 1))))
{

int sum1 = 0, sum2 = 0, magnitude;

sum1 = val1 - val2
+ 2 * input[width * (j)     + (i + 1)] - 2 * input[width * (j)     + (i - 1)]
+     val3 - val4;

sum2 = val2 + 2 * input[width * (j - 1) + (i)] + val1
- val4 - 2 * input[width * (j + 1) + (i)] - val3;

magnitude = sum1 * sum1 + sum2 * sum2;
if(magnitude > thresh)
output[index] = 255;
else
output[index] = 0;
}
else {
output[index] = 0;
}
}