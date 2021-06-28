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
__global__ void cuComputeDistanceGlobal( float* A, int wA, int pA, float* B, int wB, int pB, int dim,  float* AB){

// Declaration of the shared memory arrays As and Bs used to store the sub-matrix of A and B
__shared__ float shared_A[BLOCK_DIM][BLOCK_DIM];
__shared__ float shared_B[BLOCK_DIM][BLOCK_DIM];

// Sub-matrix of A (begin, step, end) and Sub-matrix of B (begin, step)
__shared__ int begin_A;
__shared__ int begin_B;
__shared__ int step_A;
__shared__ int step_B;
__shared__ int end_A;

// Thread index
int tx = threadIdx.x;
int ty = threadIdx.y;

// Other variables
float tmp;
float ssd = 0;

// Loop parameters
begin_A = BLOCK_DIM * blockIdx.y;
begin_B = BLOCK_DIM * blockIdx.x;
step_A  = BLOCK_DIM * pA;
step_B  = BLOCK_DIM * pB;
end_A   = begin_A + (dim-1) * pA;

// Conditions
int cond0 = (begin_A + tx < wA); // used to write in shared memory
int cond1 = (begin_B + tx < wB); // used to write in shared memory & to computations and to write in output matrix
int cond2 = (begin_A + ty < wA); // used to computations and to write in output matrix

// Loop over all the sub-matrices of A and B required to compute the block sub-matrix
for (int a = begin_A, b = begin_B; a <= end_A; a += step_A, b += step_B) {

// Load the matrices from device memory to shared memory; each thread loads one element of each matrix
if (a/pA + ty < dim){
shared_A[ty][tx] = (cond0)? A[a + pA * ty + tx] : 0;
shared_B[ty][tx] = (cond1)? B[b + pB * ty + tx] : 0;
}
else{
shared_A[ty][tx] = 0;
shared_B[ty][tx] = 0;
}

// Synchronize to make sure the matrices are loaded
__syncthreads();

// Compute the difference between the two matrixes; each thread computes one element of the block sub-matrix
if (cond2 && cond1){
for (int k = 0; k < BLOCK_DIM; ++k){
tmp = shared_A[k][ty] - shared_B[k][tx];
ssd += tmp*tmp;
}
}

// Synchronize to make sure that the preceding computation is done before loading two new sub-matrices of A and B in the next iteration
__syncthreads();
}

// Write the block sub-matrix to device memory; each thread writes one element
if (cond2 && cond1)
AB[ (begin_A + ty) * pB + begin_B + tx ] = ssd;
}