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
#define DEBUG                          0

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
* K nearest neighbor algorithm
* - Initialize CUDA
* - Allocate device memory
* - Copy point sets (reference and query points) from host to device memory
* - Compute the distances + indexes to the k nearest neighbors for each query point
* - Copy distances from device to host memory
*
* @param ref_host      reference points ; pointer to linear matrix
* @param ref_nb        number of reference points ; width of the matrix
* @param query_host    query points ; pointer to linear matrix
* @param query_nb      number of query points ; width of the matrix
* @param dim           dimension of points ; height of the matrices
* @param k             number of neighbor to consider
* @param dist_host     distances to k nearest neighbors ; pointer to linear matrix
* @param dist_host     indexes of the k nearest neighbors ; pointer to linear matrix
*
*/
__global__ void cuParallelSqrt(float *dist, int width, int k){
unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
if (xIndex<width && yIndex<k)
dist[yIndex*width + xIndex] = sqrt(dist[yIndex*width + xIndex]);
}