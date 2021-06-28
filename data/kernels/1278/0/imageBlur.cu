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





__global__ void imageBlur(float *input, float *output, int width, int height) {

int col = threadIdx.x + blockIdx.x * blockDim.x;
int row = threadIdx.y + blockIdx.y * blockDim.y;

int numcols = WIDTH;

float blur;

if (row <= height && col <= width && row > 0 && col > 0)
{
// weights
int		x1,
x3, x4, x5,
x7;

// blur
// 0.0 0.2 0.0
// 0.2 0.2 0.2
// 0.0 0.2 0.0

x1 = input[(row + 1) * numcols + col];			// up
x3 = input[row * numcols + (col - 1)];			// left
x4 = input[row * numcols + col];				// center
x5 = input[row * numcols + (col + 1)];			// right
x7 = input[(row + -1) * numcols + col];			// down

blur = (x1 * 0.2) + (x3 * 0.2) + (x4 * 0.2) + (x5 * 0.2) + (x7 * 0.2);

output[row * numcols + col] = blur;
}
}