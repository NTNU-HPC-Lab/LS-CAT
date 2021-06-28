#include "includes.h"
__global__ void convolutionParallel(unsigned char* image, unsigned char* new_image, unsigned height, unsigned width, int thread_count, int convolution_size)
{
// process image
int offset = (blockIdx.x * blockDim.x + threadIdx.x);
int width_out = (width - convolution_size + 1);
int height_out = (height - convolution_size + 1);

//Loop over pixels of smaller image
for (int i = offset; i < width_out * height_out * 4; i += thread_count)
{
int row = i / (4*width_out);
int col = i % (4*width_out);
int reference_pixel_offset = 4 * row * width + col;
float sum = 0.0;

if (convolution_size == 3)
{
float w[9] =
{
1,	2,		-1,
2,	0.25,	-2,
1,	-2,		-1
};

for (int j = 0; j < convolution_size; j++)
for (int k = 0; k < convolution_size; k++)
sum += image[reference_pixel_offset + 4 * k + 4 * j * width] * w[j * convolution_size + k];
}

if (convolution_size == 5)
{
float w[25] =
{
0.5,	0.75,	1,		-0.75,	-0.5,
0.75,	1,		2,		-1,		-0.75,
1,		2,		0.25,	-2,		-1,
0.75,	1,		-2,		-1,		-0.75,
0.5,	0.75,	-1,		-0.75,	-0.5
};

for (int j = 0; j < convolution_size; j++)
for (int k = 0; k < convolution_size; k++)
sum += image[reference_pixel_offset + 4 * k + 4 * j * width] * w[j * convolution_size + k];

}
if (convolution_size == 7)
{
float w[49] =
{
0.25,	0.3, 	0.5, 	0.75, 	-0.5, 	-0.3, 	-0.25,
0.3,	0.5,	0.75,	1,		-0.75,	-0.5, 	-0.3,
0.5,	0.75,	1,		2,		-1,		-0.75,	-0.5,
0.75,	1,		2,		0.25,	-2,		-1, 	-0.75,
0.5,	0.75,	1,		-2,		-1,		-0.75, 	-0.5,
0.3,	0.5,	0.75,	-1,		-0.75,	-0.5, 	-0.3,
0.25, 	0.3,	0.5,	-0.75,	-0.5, 	-0.3, 	-0.25

};

for (int j = 0; j < convolution_size; j++)
for (int k = 0; k < convolution_size; k++)
sum += image[reference_pixel_offset + 4 * k + 4 * j * width] * w[j * convolution_size + k];
}

if (sum <= 0)			sum = 0;
if (sum >= 255)			sum = 255;
if ((i + 1) % 4 == 0)	sum = 255; // Set a = 255

new_image[i] = (int) sum;

}

}