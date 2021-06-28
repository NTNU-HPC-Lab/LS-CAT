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




__global__ void maxpool(float *input, float *output, const int input_size, const int filter_size) {
// input : input_matrix address
// output : output buffer address
// input_size : width, height of input matrix
// filter_size : filter_size of maxpolling
// all input, output matrices are vectorized

int col = blockDim.x * blockIdx.x + threadIdx.x;
int row = blockDim.y * blockIdx.y + threadIdx.y;

// out of bound
// CHANGE

float tmp = 0.0;
float Max = -999999.9;

for(int i = 0; i < filter_size; i++){
for(int j = 0; j < filter_size; j++){
tmp = input[(input_size*filter_size*row)+(filter_size*col)+(input_size*j)+i];
if(Max<tmp)
Max = tmp;
}
}

if(col < (input_size/filter_size) && row < (input_size/filter_size))
output[((input_size/filter_size)*row)+col] = Max;

//printf("thread_made\n");
}