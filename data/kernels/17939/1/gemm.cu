#include "includes.h"
/*
* Find BLANK and replace your own code.
* And submit report why do you replace the blank that way.
*/

/* 2015004693_YangSangheon */


#define TILE_WIDTH 24      /* set TILE_WIDTH 16 for the evaluation! */
#define MAXPOOL_INPUT_FILENAME "input.txt"
#define A_FILENAME "a.txt"
#define B_FILENAME "b.txt"
#define C_FILENAME "c.txt"

using namespace std;




__global__ void gemm(float *a, float *b, float *c, const float alpha, const float beta, float *output, const int input_size){
// a, b, c : input matrix address
// alpha, beta : input constant
// output : output buffer address
// input_size : width, height of input matrix
// all input, output matrices are vectorized

int tx = threadIdx.x, ty = threadIdx.y;
int bx = blockIdx.x,  by = blockIdx.y;

int row = by*blockDim.y + ty;
int col = bx*blockDim.x + tx;

//if(row>=input_size ||col>=input_size) { return; }

if(row >= (input_size/TILE_WIDTH+1)*TILE_WIDTH ||col >= (input_size/TILE_WIDTH+1)*TILE_WIDTH) {return;}

// allocate 2D tiles in __shared__ memory
__shared__ float s_a[TILE_WIDTH][TILE_WIDTH];
__shared__ float s_b[TILE_WIDTH][TILE_WIDTH];

float result = 0;

// make sure you handle the case when the matrix sizes are not
// multiple of TILE_WIDTH!
// loop over the tiles of the input in phases

int a_index;
int b_index;

for(int p = 0; p < input_size/TILE_WIDTH+1 ;p++){
// CHANGE
// You need to use __syncthreads() a few times
// to synchronize the threads in a thread block.
a_index = row*input_size + p*TILE_WIDTH +tx;
b_index = (ty + p*TILE_WIDTH)*input_size + col;

if(a_index < input_size * input_size )
s_a[ty][tx] = a[a_index];
else
s_a[ty][tx] = 0.0;

if(b_index < input_size*input_size )
s_b[ty][tx] = b[b_index];
else
s_b[ty][tx] = 0.0;

//		s_a[ty][tx] = a[row*input_size + p*TILE_WIDTH+tx];
//		s_b[ty][tx] = b[(ty+p*TILE_WIDTH)*input_size + col];

__syncthreads();

for(int i = 0; i<TILE_WIDTH; i++)
result += s_a[ty][i] * s_b[i][tx];

__syncthreads();

}
//__syncthreads();
// write out the result to output[row*input_size + col]
// CHANGE
if(row < input_size && col < input_size)
output[row*input_size + col] = (alpha * result) + (beta * c[row*input_size + col]);
//__syncthreads();
}