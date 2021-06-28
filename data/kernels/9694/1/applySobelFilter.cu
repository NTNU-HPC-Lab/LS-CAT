#include "includes.h"
using namespace std;

#define GAUSS_WIDTH 5
#define SOBEL_WIDTH 3

typedef struct images {
char *pType;
int width;
int height;
int maxValColor;
unsigned char *data;
} image;

/**
Reads the input file formatted as pnm. The actual implementation
supports only P5 type pnm images (grayscale).
*/
__global__ void applySobelFilter(unsigned char *in, unsigned char *intensity, float *direction, int ih, int iw) {

int x = (blockIdx.x * blockDim.x) + threadIdx.x;
int y = (blockIdx.y * blockDim.y) + threadIdx.y;

int gx, gy;

if (x > 0 && x + 1 < iw && y > 0 && y + 1 < ih) {
gx =
1 * in[(y - 1) * iw + (x - 1)] + (-1) * in[(y - 1) * iw + (x + 1)] +
2 * in[y * iw + (x - 1)]	   + (-2) * in[y * iw + (x + 1)] +
1 * in[(y + 1) * iw + (x - 1)] + (-1) * in[(y + 1) * iw + (x + 1)];

gy =
1 * in[(y - 1) * iw + (x - 1)] +    2 * in[(y - 1) * iw + x] +    1 * in[(y - 1) * iw + (x + 1)] +
(-1) * in[(y + 1) * iw + (x - 1)] + (-2) * in[(y + 1) * iw + x] + (-1) * in[(y + 1) * iw + (x + 1)];

intensity[y * iw + x] = (unsigned char)sqrt((float)(gx) * (float)(gx) + (float)(gy) * (float)(gy));
direction[y * iw + x] = atan2((float)gy, (float)gx);
}
}