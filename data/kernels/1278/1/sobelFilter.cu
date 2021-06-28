#include "includes.h"


#ifdef __INTELLISENSE__
void __syncthreads();
#endif


// image dimensions WIDTH & HEIGHT
#define WIDTH 256
#define HEIGHT 256

// Block width WIDTH & HEIGHT
#define BLOCK_W 16
#define BLOCK_H 16

// buffer to read image into
float image[HEIGHT][WIDTH];

// buffer for resulting image
float final[HEIGHT][WIDTH];

// prototype declarations
void load_image();
void call_kernel();
void save_image();

#define MAXLINE 128

float total, sobel;
cudaEvent_t start_total, stop_total;
cudaEvent_t start_sobel, stop_sobel;





__global__ void sobelFilter(float *input, float *output, int width, int height) {

int col = threadIdx.x + blockIdx.x * blockDim.x;
int row = threadIdx.y + blockIdx.y * blockDim.y;

int numcols = WIDTH;

float gradient_h;
float gradient_v;
float gradient;
float thresh = 30;

if (row <= height && col <= width && row > 0 && col > 0)
{
int x0, x1, x2,
x3,	    x5,
x6, x7, x8;

// horizontal
// -1  0  1
// -2  0  2
// -1  0  1

// vertical
// -1 -2 -1
//  0  0  0
//  1  2  1

x0 = input[(row - 1) * numcols + (col - 1)];	// leftup
x1 = input[(row + 1) * numcols + col];			// up
x2 = input[(row - 1) * numcols + (col + 1)];	// rightup
x3 = input[row * numcols + (col - 1)];			// left
x5 = input[row * numcols + (col + 1)];			// right
x6 = input[(row + 1) * numcols + (col - 1)];	// leftdown
x7 = input[(row + -1) * numcols + col];			// down
x8 = input[(row + 1) * numcols + (col + 1)];	// rightdown


gradient_h = (x0 * -1) + (x2 * 1) + (x3 * -2) + (x5 * 2) + (x6 * -1) + (x8 * 1);
gradient_v = (x0 * -1) + (x1 * -2) + (x3 * -1) + (x6 * 1) + (x7 * 2) + (x8 * 1);

gradient = sqrt((gradient_h * gradient_h) + (gradient_v * gradient_v));

if (gradient >= thresh)
{
gradient = 255;
}
else {
gradient = 0;
}
output[row * numcols + col] = gradient;
}
}