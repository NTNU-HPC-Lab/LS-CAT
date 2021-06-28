#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "Matrix.h" // Header file of Matrix class

#define TILE_SIZE 16 //Used by multiplication kernel
#define BLOCK_SIZE 16 //For 2D grids it consists of 256 threads

//Matrix addition kernel
//Save sum of matrices termA and termB into result
//Matrices are passed as pointers to one dimension array
//All matrices must have the same size
//Matrices are interpreted as one-dimensional array for better performance
//Kernel uses stride pattern
__global__ void MatrixAdd(const float *termA, const float *termB,  float *result, const int size);

//Matrix subtraction kernel
//Save difference of matrices termA and termB into result
//Matrices are passed as pointers to one dimension array
//All matrices must have the same size
__global__ void MatrixSubtract(const float *termA, const float *termB,  float *result, const int size);

//Matrix multiplication kernel
//Save difference of matrices termA and termB into result
//Matrices are passed as pointers to one dimension array
//X is the width of factA and result
//Y is the height of factA and width of factB
//Z is the height of factB and result
//Width of factA must be the same as height of factB
//result sizes are factA's width x factB's height
//Uses tiling pattern
__global__ void MatrixMultiply(const float *factA, const float *factB,  float *result, const int X, const int Y, const int Z);

//Matrix transposition kernel
//Save transposed in matrix into out
//Matrices are passed as pointers to one dimension array
//out matrix must have size of inHeight x inWidth
__global__ void MatrixTranspose(const float *in, float *out, const int inWidth, const int inHeight);

//Vector elements summation kernel
//Save sum of vector elements into out
//Vector is passed as pointer to one dimension array
//out is passed as pointer to single value
//double is used for greater precision
//Uses reduction pattern
__global__ void VectorSum(float *in, double *out, int size);
