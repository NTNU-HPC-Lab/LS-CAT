#include "includes.h"
__global__ void sobel( int width_d, int height_d, int threshold_d, unsigned int *pic_d , int *final_res)
{


int row_1 = blockIdx.y * blockDim.y + threadIdx.y;

int col_1 = blockIdx.x * blockDim.x + threadIdx.x;

int tx = threadIdx.y;

int ty = threadIdx.x;

int width_Tile = TILE_SIZE;

int id, id1;

__shared__ int sharedTile[TILE_SIZE * TILE_SIZE];

int magnitude, sum1, sum2;

// Shared Tile Initialization
sharedTile[tx * width_Tile + ty]  = 0;

__syncthreads();

// Copying Data from Global to Shared Memory
sharedTile[tx * width_Tile + ty] = pic_d[row_1 * (width_d) + col_1];

__syncthreads();

// Output
if ((row_1 < height_d) && (col_1 < width_d))
{

final_res[row_1 * width_d + col_1] = 0;

}

__syncthreads();


if (row_1 > 0 && col_1 > 0 && row_1 < height_d - 1 && col_1 < width_d - 1)
{
// Applying Sobel Filter on the Tile Stored in the Shared Memory
if ((tx > 0) && (tx < width_Tile - 1)  && (ty > 0) && (ty < width_Tile - 1))
{
id = row_1 * width_d + col_1;

sum1 = sharedTile[ width_Tile * (tx-1) + ty+1] - sharedTile[ width_Tile * (tx-1) + ty-1 ] + 2 * sharedTile[ width_Tile * (tx)   + ty+1 ] - 2 * sharedTile[ width_Tile*(tx)   + ty-1 ] +  sharedTile[ width_Tile * (tx+1) + ty+1] - sharedTile[ width_Tile*(tx+1) + ty-1 ];

sum2 = sharedTile[ width_Tile * (tx-1) + ty-1 ] + 2 * sharedTile[ width_Tile * (tx-1) + ty ] + sharedTile[ width_Tile * (tx-1) + ty+1] - sharedTile[width_Tile * (tx+1) + ty-1 ] - 2 * sharedTile[ width_Tile * (tx+1) + ty ] - sharedTile[ width_Tile * (tx+1) + ty+1];

magnitude = sum1 * sum1 + sum2 * sum2;

if (magnitude > threshold_d)

{

final_res[id] = 255;

}

else

{

final_res[id] = 0;

}

}

__syncthreads();

// For the Pixels at the Boundaries of the Block using Global Memory

if ((row_1 == blockIdx.y * blockDim.y + blockDim.y - 1) || (col_1 == blockIdx.x * blockDim.x + blockDim.x - 1) || (row_1 == blockIdx.y * blockDim.y) || (col_1 == blockIdx.x * blockDim.x))

{
id1 = row_1 * width_d + col_1;

sum1 =  pic_d[ width_d * (row_1-1) + col_1+1] - pic_d[ width_d * (row_1-1) + col_1-1 ] + 2 * pic_d[ width_d * (row_1) + col_1+1 ] - 2 * pic_d[ width_d*(row_1)   + col_1-1 ] + pic_d[ width_d * (row_1+1) + col_1+1] - pic_d[ width_d*(row_1+1) + col_1-1 ];

sum2 = pic_d[ width_d * (row_1-1) + col_1-1 ] + 2 * pic_d[ width_d * (row_1-1) + col_1 ] + pic_d[ width_d * (row_1-1) + col_1+1] - pic_d[width_d * (row_1+1) + col_1-1 ] - 2 * pic_d[ width_d * (row_1+1) + col_1 ] - pic_d[ width_d * (row_1+1) + col_1+1];



magnitude =  sum1*sum1 + sum2*sum2;

if (magnitude > threshold_d)

{

final_res[id1] = 255;

}

else

{

final_res[id1] = 0;

}

}

__syncthreads();

}

}