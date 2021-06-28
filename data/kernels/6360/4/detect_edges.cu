#include "includes.h"
__global__ void detect_edges(unsigned char *in, unsigned char *out) {
int i;
int n_pixels = width * height;

for(i=0;i<n_pixels;i++) {
int x, y; // the pixel of interest
int b, d, f, h; // the pixels adjacent to x,y used for the calculation
int r; // the result of calculate

y = i / width;
x = i - (width * y);

if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
out[i] = 0;
} else {
b = i + width;
d = i - 1;
f = i + 1;
h = i - width;

r = (in[i] * 4) + (in[b] * -1) + (in[d] * -1) + (in[f] * -1)
+ (in[h] * -1);

if (r > 0) { // if the result is positive this is an edge pixel
out[i] = 255;
} else {
out[i] = 0;
}
}
}
}