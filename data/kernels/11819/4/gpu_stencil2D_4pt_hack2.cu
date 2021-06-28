#include "includes.h"
extern "C" {
}
#define ROTATE_DOWN(val,MAX) ((val-1==-1)?MAX-1:val-1)
#define ROTATE_UP(val,MAX) ((val+1)%MAX)
/**
* GPU Device kernel for the for 2D stencil
* First attempt during hackaton
* M = Rows, N = Cols INCLUDING HALOS
* In this version now we replace the size of the shared memory to be just 3 rows (actually 1+HALO*2) rows
*/

__global__ void gpu_stencil2D_4pt_hack2(double * dst, double * src, int M, int N)
{
//	printf("kernel begin!\n");
//Declaring the shared memory array for source
__shared__ double shared_mem[ 1 + HALO*2 ] [ GRID_TILE_X + HALO*2]; //1 is the row I am modifying
//double * shSrc = shared_mem;

//indexes
int i, j, curRow;
//Cols   *  numRows/Tile * tileIndex
int base_global_idx = ( N ) * ( GRID_TILE_Y * blockIdx.y ) + GRID_TILE_X*blockIdx.x;

int center = 1,north = 0,south = 2; //indexes for the current location in the shared memory

//copy the shared memory to fill the pipeline
for (i = 0 ; i < 1+HALO*2 ; i ++ )
for (j = threadIdx.x ; j < GRID_TILE_X+2*HALO ; j+=blockDim.x)
{
shared_mem [i][j] = src[base_global_idx + i*N + j];
}
__syncthreads();
//Pipelined copy one row and process it
for ( curRow = HALO; curRow < GRID_TILE_Y; curRow+=1 )
{
//Stencil computation
for (j = threadIdx.x + HALO ; j < GRID_TILE_X+HALO ; j+=blockDim.x)
{
//top             + bottom              + left                + right
dst[base_global_idx + curRow*N + j] = (shared_mem[north][j] + shared_mem[south][j] + shared_mem[center][j-1] + shared_mem[center][j+1] )/5.5;
}

__syncthreads();
//We are copying from dst to shared memory.
for (j = threadIdx.x ; j < GRID_TILE_X+2*HALO ; j+=blockDim.x)
{
shared_mem [north][j] = src[base_global_idx + (curRow+2)*N + j];
}

center = ROTATE_UP(center,3);
south  = ROTATE_UP(south,3);
north = ROTATE_UP(north,3);
__syncthreads();
}

//Dranning the pipeline
for (j = threadIdx.x + HALO ; j < GRID_TILE_X+HALO ; j+=blockDim.x)
{
//top             + bottom              + left                + right
dst[base_global_idx + curRow*N + j] = (shared_mem[north][j] + shared_mem[south][j] + shared_mem[center][j-1] + shared_mem[center][j+1] )/5.5;
}
__syncthreads();

//	printf("kernel finish!\n");
}