#include "includes.h"

#define BLOCK_SIZE 1
//Global variable set up
const int radius = 3;
const int numSamples = 100;
const double learningRate = 0.15;
const int epochs = 1;
const int numNeurons =20;
//Set up neurons
double inputLayer[2][numNeurons] = { 0 }; //takes input and weights
double outputLayer[1][numNeurons] = { 0 }; //takes weights and outputs
double * matrixA; //Temporarily initialised arrays which are allocated to aid in gpu memory allocation
double * matrixB;
double * matrixC;
double * matrixD;
double * matrixE;

//Calculates dot product of two arrays from a given pointer and returns a total - must be same size
__global__ void MatMultKernel(double *array1, double *array2, double *output, int arr1_rows, int arr1_cols, int arr2_cols) {
double result = 0;
__shared__ double subArray1[BLOCK_SIZE][BLOCK_SIZE]; //Setting up the shared memory into sub arrays for more efficient computation
__shared__ double subArray2[BLOCK_SIZE][BLOCK_SIZE];
int bIDx = blockIdx.x, bIDy = blockIdx.y, tIDx = threadIdx.x, tIDy = threadIdx.y; //Setting up variables to identify threads uniquely
int row = bIDy * BLOCK_SIZE + tIDy; //Setting the given row of a thread
int col = bIDx * BLOCK_SIZE + tIDx; //Setting the given col of a thread
for (int i = 0; i < (arr1_cols-1)/BLOCK_SIZE+1; i++) { //Iterating through every chunk of columns proportional to block size
if (row < arr1_rows && i*BLOCK_SIZE+tIDx<arr1_cols) {
subArray1[tIDy][tIDx] = array1[row*arr1_cols + i * BLOCK_SIZE + tIDx]; //Setting up sub array1 to contain relevant pieces of array1
}else {
subArray1[tIDy][tIDx] = 0; //0ing values to prevent miscalculation if not relevant
}
if (col < arr2_cols && i*BLOCK_SIZE+tIDy<arr1_cols) {
subArray2[tIDy][tIDx] = array2[(i * BLOCK_SIZE + tIDy)*arr2_cols+col]; //Setting up sub array2 to contain relevant pieces of array2
}else {
subArray2[tIDy][tIDx] = 0;//0ing values to prevent miscalculation if not relevant
}
__syncthreads(); //Blocking to ensure sub arrays are built
for (int ii = 0; ii < BLOCK_SIZE; ii++) {
result += subArray1[tIDy][ii] * subArray2[ii][tIDx]; //Calculating result for this chunk utilising many threads simultaneously
}
__syncthreads(); //Ensure result calculation is done
}
if (row < arr1_rows&&col < arr2_cols) {
output[row*arr2_cols + col] = result; //Calculate overall output in position
}

}