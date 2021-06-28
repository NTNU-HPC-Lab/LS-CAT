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
__global__ void extract_with_interpolation( int nthreads, float *data, float *n_xy_coords, float *extracted_data, int n_max_coord, int channels, int height, int width) {

int x0, x1, y0, y1, nc;
float wx0, wx1, wy0, wy1;
int n, nd;
float x, y;

for (int index = blockIdx.x * blockDim.x + threadIdx.x;
index < (nthreads);
index += blockDim.x * gridDim.x) {
n = (index / n_max_coord);
nd = n * n_max_coord * channels;
x = n_xy_coords[index * 2];
y = n_xy_coords[index * 2 + 1];

x0 = static_cast<int>(floor(x));
x1 = x0 + 1;
y0 = static_cast<int>(floor(y));
y1 = y0 + 1;

x0 = x0 <= 0 ? 0 : (x0 >= (width - 1)  ? (width - 1) : x0);
y0 = y0 <= 0 ? 0 : (y0 >= (height - 1) ? (height - 1) : y0);
x1 = x1 <= 0 ? 0 : (x1 >= (width - 1)  ? (width - 1) : x1);
y1 = y1 <= 0 ? 0 : (y1 >= (height - 1) ? (height - 1) : y1);

wx0 = static_cast<float>(x1) - x;
wx1 = x - x0;
wy0 = static_cast<float>(y1) - y;
wy1 = y - y0;

if(x0 == x1){ wx0 = 1; wx1 = 0; }
if(y0 == y1){ wy0 = 1; wy1 = 0; }
for(int c=0; c < channels; c++) {
nc = (n * channels + c) * height;
// extracted_data[index * channels + c] = wy0 * wx0 * data[(nc + y0) * width + x0]
// extracted_data[nd + index % n_max_coord + n_max_coord * c] = index;
extracted_data[nd + index % n_max_coord + n_max_coord * c] = wy0 * wx0 * data[(nc + y0) * width + x0]
+ wy1 * wx0 * data[(nc + y1) * width + x0]
+ wy0 * wx1 * data[(nc + y0) * width + x1]
+ wy1 * wx1 * data[(nc + y1) * width + x1];
}
}
}