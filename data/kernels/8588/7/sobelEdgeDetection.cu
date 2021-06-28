#include "includes.h"
__global__ void sobelEdgeDetection(int *input, int *output, int width, int height, int thresh) {

int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;
int index = j * width + i;

if ( ((i > 0) && (j > 0)) && ((i < (width - 1)) && (j < (height - 1))))
{

int sum1 = 0, sum2 = 0, magnitude;

sum1 = input[width * (j - 1) + (i + 1)] -     input[width * (j - 1) + (i - 1)]
+ 2 * input[width * (j)     + (i + 1)] - 2 * input[width * (j)     + (i - 1)]
+     input[width * (j + 1) + (i + 1)] -     input[width * (j + 1) + (i - 1)];

sum2 = input[width * (j - 1) + (i - 1)] + 2 * input[width * (j - 1) + (i)] + input[width * (j - 1) + (i + 1)]
- input[width * (j + 1) + (i - 1)] - 2 * input[width * (j + 1) + (i)] - input[width * (j + 1) + (i + 1)];

magnitude = sum1 * sum1 + sum2 * sum2;
if(magnitude > thresh)
output[index] = 255;
else
output[index] = 0;
}
}