#include "includes.h"
//header files included

//declaring the tile width and height
//for tile based matrix multiplication
#define TILE_WIDTH 32
#define TILE_HEIGHT 32

//Namespace for std
using namespace std;

//structure declaration for storing rows and columns for a matrix
struct matrix{
unsigned int rows;	//storing rows of a matrix
unsigned int cols;	//storing columns of a matrix
};

//handlerror declaration : to display file and line numbers of erroneous lines
__global__ void matrix_mult(float* array1, unsigned int rows1, unsigned int cols1, float* array2, unsigned int rows2, unsigned int cols2, float* array3)
{
//shared memory takes one tile at a time
__shared__ float S1[TILE_WIDTH][TILE_HEIGHT];	//to store tiles for array 1
__shared__ float S2[TILE_HEIGHT][TILE_WIDTH];	//to store tiles for array 2

//threads x and y index for the current block
unsigned int tx=threadIdx.x;
unsigned int ty=threadIdx.y;

unsigned int c=blockIdx.x*blockDim.x + threadIdx.x;	//row value using x-index of current thread
unsigned int r=blockIdx.y*blockDim.y + threadIdx.y;	//column value using y-index of current thread

unsigned int idx=c*rows1+r;				//column major index, using row and column value

float val=0;		//register to store multiplication result initialized to zero

for(int m=0; m<1+((rows2-1)/TILE_WIDTH);m++)	//going over all tiles one by one, with each m
{

int var1=m*TILE_WIDTH+tx ;		//x thread value for current tile
int var2=m*TILE_WIDTH+ty ;		//y thread value for current tile

//copying a tile from array1
if (r < rows1 && var1 < rows2)		//if the value is associated to a valid matrix coordinate in array1 then store it to shared memory S1
S1[ty][tx]=array1[r + var1*rows1];//storing a "valid" value from array to shared memory
else
S1[ty][tx]=0;					//storing zero, since there is no valid value
__syncthreads();						//syncing all threads once shared memory S1 is stored

//copying a tile from array2
if(c < cols2 && var2 < rows2)	//if value is associates to a valid matrix coordinate in array2 then store it to shared memory S2
S2[ty][tx]=array2[var2+rows2*c];	//storing the valid value
else
S2[ty][tx]=0;		//storing zero, since no valid value
__syncthreads();		//synchronizing threads


for(int i=0; i<TILE_WIDTH;i++)	//going over entire tile, ty row in S1 and tx column in S2
val+=S1[ty][i]*S2[i][tx];	//and multiplying elements
__syncthreads();		//synchronizing threads

}

if(r < rows1 && c< cols2)	//removing degenerate cases
array3[idx]=val;	//saving multiplication result to global memory

}