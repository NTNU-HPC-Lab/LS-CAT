#include "includes.h"
/******************************
*STUDENT NAME: DAVID PARKS    *
*PROJECT: 6 - GREY SCALE FLIP *
*DUE DATE: THURS 18/10/18     *
*******************************/

#define PPM_MAGIC_1 'P'
#define PPM_MAGIC_2 '6'
#define BLOCK_SIZE 16;

struct PPM_header {
int width;
int height;
int max_color;
};
struct RGB_8 {
uint8_t r;
uint8_t g;
uint8_t b;
};//__attribute__((packed));

__global__ void gray_scale_flip(RGB_8* img, int height, int width)
{
int row = blockDim.y * blockIdx.y + threadIdx.y;
int col = blockDim.x * blockIdx.x + threadIdx.x;

if (row < height && col < width / 2)
{
int i = row * width + col;
//temp var for slip pixel
RGB_8 temp = img[(row + 1) * width - col - 1];

//computing gray value
float gray_value = 0.21 * img[i].r + 0.72 * img[i].g + 0.07 * img[i].b;
img[i].r = gray_value;
img[i].g = gray_value;
img[i].b = gray_value;

//set flip pixel to grayed current pixel
img[(row + 1) * width - col - 1] = img[i];

//set current pixel to temp pixel
img[i] = temp;

//computing gray value
gray_value = 0.21 * img[i].r + 0.72 * img[i].g + 0.07 * img[i].b;
img[i].r = gray_value;
img[i].g = gray_value;
img[i].b = gray_value;
}
}