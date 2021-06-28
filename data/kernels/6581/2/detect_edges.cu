#include "includes.h"
__global__ void detect_edges(unsigned char *input, unsigned char *output) {
int i = (blockIdx.x * 72) + threadIdx.x;
int x, y; // the pixel of interest
int b, d, f, h; // the pixels adjacent to the x,y used to calculate
int r; // the calculation result
y = i / width;;
x = i - (width * y);
if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
output[i] = 0;
} else {
b = i + width;
d = i - 1;
f = i + 1;
h = i - width;
r = (input[i] * 4) + (input[b] * -1) + (input[d] * -1) + (input[f] * -1)
+ (input[h] * -1);
if (r >= 0) {
output[i] = 0;
} else {
output[i] = 255;
}
}
}