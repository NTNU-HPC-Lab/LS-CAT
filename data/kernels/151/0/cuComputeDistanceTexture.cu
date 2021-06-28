#include "includes.h"
/**
*
* Date         03/07/2009
* ====
*
* Authors      Vincent Garcia
* =======      Eric    Debreuve
*              Michel  Barlaud
*
* Description  Given a reference point set and a query point set, the program returns
* ===========  the distance between each query point and its k-th nearest neighbor in
*              the reference point set. Only the distance is provided. The computation
*              is performed using the API NVIDIA CUDA.
*
* Paper        Fast k nearest neighbor search using GPU
* =====
*
* BibTeX       @INPROCEEDINGS{2008_garcia_cvgpu,
* ======         author = {V. Garcia and E. Debreuve and M. Barlaud},
*                title = {Fast k nearest neighbor search using GPU},
*                booktitle = {CVPR Workshop on Computer Vision on GPU},
*                year = {2008},
*                address = {Anchorage, Alaska, USA},
*                month = {June}
*              }
*
*/


// If the code is used in Matlab, set MATLAB_CODE to 1. Otherwise, set MATLAB_CODE to 0.
#define MATLAB_CODE 0


// Includes
#if MATLAB_CODE == 1
#else
#endif


// Constants used by the program
#define MAX_PITCH_VALUE_IN_BYTES       262144
#define MAX_TEXTURE_WIDTH_IN_BYTES     65536
#define MAX_TEXTURE_HEIGHT_IN_BYTES    32768
#define MAX_PART_OF_FREE_MEMORY_USED   0.9
#define BLOCK_DIM                      16


// Texture containing the reference points (if it is possible)
texture<float, 2, cudaReadModeElementType> texA;



//-----------------------------------------------------------------------------------------------//
//                                            KERNELS                                            //
//-----------------------------------------------------------------------------------------------//



/**
* Computes the distance between two matrix A (reference points) and
* B (query points) containing respectively wA and wB points.
* The matrix A is a texture.
*
* @param wA    width of the matrix A = number of points in A
* @param B     pointer on the matrix B
* @param wB    width of the matrix B = number of points in B
* @param pB    pitch of matrix B given in number of columns
* @param dim   dimension of points = height of matrices A and B
* @param AB    pointer on the matrix containing the wA*wB distances computed
*/


/**
* Computes the distance between two matrix A (reference points) and
* B (query points) containing respectively wA and wB points.
*
* @param A     pointer on the matrix A
* @param wA    width of the matrix A = number of points in A
* @param pA    pitch of matrix A given in number of columns
* @param B     pointer on the matrix B
* @param wB    width of the matrix B = number of points in B
* @param pB    pitch of matrix B given in number of columns
* @param dim   dimension of points = height of matrices A and B
* @param AB    pointer on the matrix containing the wA*wB distances computed
*/



/**
* Gathers k-th smallest distances for each column of the distance matrix in the top.
*
* @param dist     distance matrix
* @param width    width of the distance matrix
* @param pitch    pitch of the distance matrix given in number of columns
* @param height   height of the distance matrix
* @param k        number of smallest distance to consider
*/



/**
* Computes the square root of the first line (width-th first element)
* of the distance matrix.
*
* @param dist    distance matrix
* @param width   width of the distance matrix
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
__global__ void cuComputeDistanceTexture(int wA, float * B, int wB, int pB, int dim, float* AB){
unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
if ( xIndex<wB && yIndex<wA ){
float ssd = 0;
for (int i=0; i<dim; i++){
float tmp  = tex2D(texA, (float)yIndex, (float)i) - B[ i * pB + xIndex ];
ssd += tmp * tmp;
}
AB[yIndex * pB + xIndex] = ssd;
}
}