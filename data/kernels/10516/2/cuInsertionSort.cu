#include "includes.h"
/** Modifed version of knn-CUDA from https://github.com/vincentfpgarcia/kNN-CUDA
* The modifications are
*      removed texture memory usage
*      removed split query KNN computation
*      added feature extraction with bilinear interpolation
*
* Last modified by Christopher B. Choy <chrischoy@ai.stanford.edu> 12/23/2016
*/

// Includes

// Constants used by the program
#define BLOCK_DIM                      16


//-----------------------------------------------------------------------------------------------//
//                                            KERNELS                                            //
//-----------------------------------------------------------------------------------------------//

/**
* Computes the distance between two matrix A (reference points) and
* B (query points) containing respectively wA and wB points.
*
* @param A     pointer on the matrix A
* @param wA    width of the matrix A = number of points in A
* @param B     pointer on the matrix B
* @param wB    width of the matrix B = number of points in B
* @param dim   dimension of points = height of matrices A and B
* @param AB    pointer on the matrix containing the wA*wB distances computed
*/


/**
* Gathers k-th smallest distances for each column of the distance matrix in the top.
*
* @param dist        distance matrix
* @param ind         index matrix
* @param width       width of the distance matrix and of the index matrix
* @param height      height of the distance matrix and of the index matrix
* @param k           number of neighbors to consider
*/


/**
* Computes the square root of the first line (width-th first element)
* of the distance matrix.
*
* @param dist    distance matrix
* @param width   width of the distance matrix
* @param k       number of neighbors to consider
*/


//-----------------------------------------------------------------------------------------------//
//                                   K-th NEAREST NEIGHBORS                                      //
//-----------------------------------------------------------------------------------------------//


/**
* Prints the error message return during the memory allocation.
*
* @param error        error value return by the memory allocation function
* @param memorySize   size of memory tried to be allocated
*/
__global__ void cuInsertionSort(float *dist, int *ind, int width, int height, int k){

// Variables
int l, i, j;
float *p_dist;
int   *p_ind;
float curr_dist, max_dist;
int   curr_row,  max_row;
unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

if (xIndex<width){
// Pointer shift, initialization, and max value
p_dist   = dist + xIndex;
p_ind    = ind  + xIndex;
max_dist = p_dist[0];
p_ind[0] = 1;

// Part 1 : sort kth firt elementZ
for (l=1; l<k; l++){
curr_row  = l * width;
curr_dist = p_dist[curr_row];
if (curr_dist<max_dist){
i=l-1;
for (int a=0; a<l-1; a++){
if (p_dist[a*width]>curr_dist){
i=a;
break;
}
}
for (j=l; j>i; j--){
p_dist[j*width] = p_dist[(j-1)*width];
p_ind[j*width]   = p_ind[(j-1)*width];
}
p_dist[i*width] = curr_dist;
p_ind[i*width]   = l+1;
} else {
p_ind[l*width] = l+1;
}
max_dist = p_dist[curr_row];
}

// Part 2 : insert element in the k-th first lines
max_row = (k-1)*width;
for (l=k; l<height; l++){
curr_dist = p_dist[l*width];
if (curr_dist<max_dist){
i=k-1;
for (int a=0; a<k-1; a++){
if (p_dist[a*width]>curr_dist){
i=a;
break;
}
}
for (j=k-1; j>i; j--){
p_dist[j*width] = p_dist[(j-1)*width];
p_ind[j*width]   = p_ind[(j-1)*width];
}
p_dist[i*width] = curr_dist;
p_ind[i*width]   = l+1;
max_dist             = p_dist[max_row];
}
}
}
}