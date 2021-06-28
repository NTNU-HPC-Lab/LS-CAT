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
__global__ void applyGaussianFilter(unsigned char *input, unsigned char *output, float *kernel, int iHeight, int iWidth, int kWidth) {

int x = (blockIdx.x * blockDim.x) + threadIdx.x;
int y = (blockIdx.y * blockDim.y) + threadIdx.y;

double sum = 0.0;

int halvedKW = kWidth / 2;

for (int i = -halvedKW; i <= halvedKW; i++) {
for (int j = -halvedKW; j <= halvedKW; j++) {
if ((x + j) < iWidth && (x + j) >= 0 && (y + i) < iHeight && (y + i) >= 0) {
int kPosX = (j + halvedKW);
int kPosY = (i + halvedKW);
sum = sum + (float)(input[(y + i) * iWidth + (x + j)]) * kernel[kPosY * kWidth + kPosX];
}
}
}

if (sum > 255.0)
sum = 255.0;

output[y * iWidth + x] = (unsigned char)sum;
}