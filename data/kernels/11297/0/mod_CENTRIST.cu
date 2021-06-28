#include "includes.h"
#define TILE_WIDTH 32

#define COMMENT "Centrist_GPU"
#define RGB_COMPONENT_COLOR 255

typedef struct {
unsigned char red, green, blue;
} PPMPixel;

typedef struct {
int x, y;
PPMPixel *data;
} PPMImage;

__global__ void mod_CENTRIST(PPMPixel *image_out, PPMPixel *image_cp, int columns, int rows, int *hist, int hist_len) {

int col = TILE_WIDTH * blockIdx.x + threadIdx.x;
int row = TILE_WIDTH * blockIdx.y + threadIdx.y;

__shared__ int hist_private[512];
int hist_index = (threadIdx.y*TILE_WIDTH + threadIdx.x); //get index in shared histogram
if(hist_index < hist_len) hist_private[hist_index] = 0;
__syncthreads();
if(col < columns && row < rows)
{
//create and copy small chunks to shared memory
__shared__ unsigned char image_cp_private[TILE_WIDTH][TILE_WIDTH];

//convert to grayscale
int img_index = row * columns + col; //get index in original image
int grayscale = (image_cp[img_index].red*299 + image_cp[img_index].green*587 + image_cp[img_index].blue*114)/1000; //avoid float point errors

image_cp_private[threadIdx.y][threadIdx.x] = grayscale;

__syncthreads();
if(col < columns - 2 && row < rows - 2) //ignore first/last row/column
{
int r, c, rr, cc;
float mean = 0.0;
for(r = threadIdx.y, rr = row; r <= threadIdx.y + 2; r++, rr++)
for(c = threadIdx.x , cc = col; c <= threadIdx.x + 2; c++, cc++)
{
if(r < TILE_WIDTH && c < TILE_WIDTH)
{
mean += image_cp_private[r][c];
}
else
{
int grayscale_neigh = (image_cp[rr*columns + cc].red*299 + image_cp[rr*columns + cc].green*587 + image_cp[rr*columns + cc].blue*114)/1000;
mean += grayscale_neigh;
}
}
mean /= 9.0;
int value = 0, k = 8;
for(r = threadIdx.y, rr = row ; r <= threadIdx.y + 2; r++, rr++)
for(c = threadIdx.x, cc = col ; c <= threadIdx.x + 2; c++, cc++)
{
if(r < TILE_WIDTH && c < TILE_WIDTH)
{
if(1.0*image_cp_private[r][c] >= mean)
value |= 1<<k;
}
else
{
int grayscale_neigh = (image_cp[rr*columns + cc].red*299 + image_cp[rr*columns + cc].green*587 + image_cp[rr*columns + cc].blue*114)/1000;
if(grayscale_neigh >= mean)
value |= 1<<k;
}
k--;
}
int img_out_ind = row * (columns - 2) + col; //get index in ouput original
image_out[img_out_ind].red = image_out[img_out_ind].blue = image_out[img_out_ind].green = value;
atomicAdd(&(hist_private[value]), 1);
}
__syncthreads();
if(hist_index == 0)
{
for(int i = 0; i < hist_len; i++)
atomicAdd(&(hist[i]), hist_private[i]); //init shared histogram
}
}
}