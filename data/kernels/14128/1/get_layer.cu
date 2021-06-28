#include "includes.h"

//Number of elements of the inpu layers, that correspond to the number of pixels of a picture
#define PIXELS 3073
//Number of elements of the first hidden layer
#define HIDDEN_LAYER_1 2000
//Number of elements of the second hidden layer
#define HIDDEN_LAYER_2 450
//Number of elements of the output layer
#define OUTPUT_LAYER 10
//Learning rate of the algorithm
#define LEARNING_RATE 0.01
//Numbers of elements to use for training
#define ELEMENTS 1000
//Blocks
#define BLOCKS 32

/*
* Function that given a vector and its size, print it
* In:
* f: vector of doubles to be printed
* N: size of the vector
*/
__global__ void get_layer(double *input, double *matrix, double *result,int input_size, int hidden_size){

int x = blockIdx.x*blockDim.x + threadIdx.x;
int y = blockIdx.y*blockDim.y + threadIdx.y;
if (x < hidden_size && y < input_size)
result[x] += input[y]*matrix[y*hidden_size+x];
}