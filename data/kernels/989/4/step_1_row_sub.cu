#include "includes.h"
// Fast Block Distributed CUDA Implementation of the Hungarian Algorithm
//
// Annex to the paper:
// Paulo A. C. Lopes, Satyendra Singh Yadav, Aleksandar Ilic, Sarat Kumar Patra ,
// "Fast Block Distributed CUDA Implementation of the Hungarian Algorithm",
// Journal Parallel Distributed Computing
//
// Hungarian algorithm:
// (This algorithm was modified to result in an efficient GPU implementation, see paper)
//
// Initialize the slack matrix with the cost matrix, and then work with the slack matrix.
//
// STEP 1: Subtract the row minimum from each row. Subtract the column minimum from each column.
//
// STEP 2: Find a zero of the slack matrix. If there are no starred zeros in its column or row star the zero.
// Repeat for each zero.
//
// STEP 3: Cover each column with a starred zero. If all the columns are
// covered then the matching is maximum.
//
// STEP 4: Find a non-covered zero and prime it. If there is no starred zero in the row containing this primed zero,
// Go to Step 5. Otherwise, cover this row and uncover the column containing the starred zero.
// Continue in this manner until there are no uncovered zeros left.
// Save the smallest uncovered value and Go to Step 6.
//
// STEP 5: Construct a series of alternating primed and starred zeros as follows:
// Let Z0 represent the uncovered primed zero found in Step 4.
// Let Z1 denote the starred zero in the column of Z0(if any).
// Let Z2 denote the primed zero in the row of Z1(there will always be one).
// Continue until the series terminates at a primed zero that has no starred zero in its column.
// Un-star each starred zero of the series, star each primed zero of the series,
// erase all primes and uncover every row in the matrix. Return to Step 3.
//
// STEP 6: Add the minimum uncovered value to every element of each covered row,
// and subtract it from every element of each uncovered column.
// Return to Step 4 without altering any stars, primes, or covered rows.


// Uncomment to use chars as the data type, otherwise use int
// #define CHAR_DATA_TYPE

// Uncomment to use a 4x4 predefined matrix for testing
// #define USE_TEST_MATRIX

// Comment to use managed variables instead of dynamic parallelism; usefull for debugging
// #define DYNAMIC

#define klog2(n) ((n<8)?2:((n<16)?3:((n<32)?4:((n<64)?5:((n<128)?6:((n<256)?7:((n<512)?8:((n<1024)?9:((n<2048)?10:((n<4096)?11:((n<8192)?12:((n<16384)?13:0))))))))))))

#ifndef DYNAMIC
#define MANAGED __managed__
#define dh_checkCuda checkCuda
#define dh_get_globaltime get_globaltime
#define dh_get_timer_period get_timer_period
#else
#define dh_checkCuda d_checkCuda
#define dh_get_globaltime d_get_globaltime
#define dh_get_timer_period d_get_timer_period
#define MANAGED
#endif

#define kmin(x,y) ((x<y)?x:y)
#define kmax(x,y) ((x>y)?x:y)

#ifndef USE_TEST_MATRIX
#ifdef _n_
// These values are meant to be changed by scripts
const int n = _n_;							// size of the cost/pay matrix
const int range = _range_;					// defines the range of the random matrix.
const int user_n = n;
const int n_tests = 100;
#else
// User inputs: These values should be changed by the user
const int user_n = 1000;				// This is the size of the cost matrix as supplied by the user
const int n = 1<<(klog2(user_n)+1);		// The size of the cost/pay matrix used in the algorithm that is increased to a power of two
const int range = n;					// defines the range of the random matrix.
const int n_tests = 10;					// defines the number of tests performed
#endif

// End of user inputs

const int log2_n = klog2(n);			// log2(n)
const int n_threads = kmin(n,64);		// Number of threads used in small kernels grid size (typically grid size equal to n)
// Used in steps 3ini, 3, 4ini, 4a, 4b, 5a and 5b (64)
const int n_threads_reduction = kmin(n, 256);		// Number of threads used in the redution kernels in step 1 and 6 (256)
const int n_blocks_reduction = kmin(n, 256);		// Number of blocks used in the redution kernels in step 1 and 6 (256)
const int n_threads_full = kmin(n, 512);			// Number of threads used the largest grids sizes (typically grid size equal to n*n)
// Used in steps 2 and 6 (512)
const int seed = 45345;					// Initialization for the random number generator

#else
const int n = 4;
const int log2_n = 2;
const int n_threads = 2;
const int n_threads_reduction = 2;
const int n_blocks_reduction = 2;
const int n_threads_full = 2;
#endif

const int n_blocks = n / n_threads;									// Number of blocks used in small kernels grid size (typically grid size equal to n)
const int n_blocks_full = n * n / n_threads_full;					// Number of blocks used the largest gris sizes (typically grid size equal to n*n)
const int row_mask = (1 << log2_n) - 1;								// Used to extract the row from tha matrix position index (matrices are column wise)
const int nrows = n, ncols = n;										// The matrix is square so the number of rows and columns is equal to n
const int max_threads_per_block = 1024;								// The maximum number of threads per block
const int columns_per_block_step_4 = 512;							// Number of columns per block in step 4
const int n_blocks_step_4 = kmax(n / columns_per_block_step_4, 1);	// Number of blocks in step 4 and 2
const int data_block_size = columns_per_block_step_4 * n;			// The size of a data block. Note that this can be bigger than the matrix size.
const int log2_data_block_size = log2_n + klog2(columns_per_block_step_4);	// log2 of the size of a data block. Note that klog2 cannot handle very large sizes

// For the selection of the data type used
#ifndef CHAR_DATA_TYPE
typedef int data;
#define MAX_DATA INT_MAX
#define MIN_DATA INT_MIN
#else
typedef unsigned char data;
#define MAX_DATA 255
#define MIN_DATA 0
#endif

// Host Variables

// Some host variables start with h_ to distinguish them from the corresponding device variables
// Device variables have no prefix.

#ifndef USE_TEST_MATRIX
data h_cost[ncols][nrows];
#else
data h_cost[n][n] = { { 1, 2, 3, 4 }, { 2, 4, 6, 8 }, { 3, 6, 9, 12 }, { 4, 8, 12, 16 } };
#endif
int h_column_of_star_at_row[nrows];
int h_zeros_vector_size;
int h_n_matches;
bool h_found;
bool h_goto_5;

// Device Variables

__device__ data slack[nrows*ncols];						// The slack matrix
__device__ data min_in_rows[nrows];						// Minimum in rows
__device__ data min_in_cols[ncols];						// Minimum in columns
__device__ int zeros[nrows*ncols];						// A vector with the position of the zeros in the slack matrix
__device__ int zeros_size_b[n_blocks_step_4];			// The number of zeros in block i

__device__ int row_of_star_at_column[ncols];			// A vector that given the column j gives the row of the star at that column (or -1, no star)
__device__ int column_of_star_at_row[nrows];			// A vector that given the row i gives the column of the star at that row (or -1, no star)
__device__ int cover_row[nrows];						// A vector that given the row i indicates if it is covered (1- covered, 0- uncovered)
__device__ int cover_column[ncols];						// A vector that given the column j indicates if it is covered (1- covered, 0- uncovered)
__device__ int column_of_prime_at_row[nrows];			// A vector that given the row i gives the column of the prime at that row  (or -1, no prime)
__device__ int row_of_green_at_column[ncols];			// A vector that given the row j gives the column of the green at that row (or -1, no green)

__device__ data max_in_mat_row[nrows];					// Used in step 1 to stores the maximum in rows
__device__ data min_in_mat_col[ncols];					// Used in step 1 to stores the minimums in columns
__device__ data d_min_in_mat_vect[n_blocks_reduction];	// Used in step 6 to stores the intermediate results from the first reduction kernel
__device__ data d_min_in_mat;							// Used in step 6 to store the minimum

MANAGED __device__ int zeros_size;					// The number fo zeros
MANAGED __device__ int n_matches;					// Used in step 3 to count the number of matches found
MANAGED __device__ bool goto_5;						// After step 4, goto step 5?
MANAGED __device__ bool repeat_kernel;				// Needs to repeat the step 2 and step 4 kernel?
#if defined(DEBUG) || defined(_DEBUG)
MANAGED __device__ int n_covered_rows;				// Used in debug mode to check for the number of covered rows
MANAGED __device__ int n_covered_columns;			// Used in debug mode to check for the number of covered columns
#endif

__shared__ extern data sdata[];							// For access to shared memory

// -------------------------------------------------------------------------------------
// Device code
// -------------------------------------------------------------------------------------

__global__ void step_1_row_sub()
{

int i = blockDim.x * blockIdx.x + threadIdx.x;
int l = i & row_mask;

slack[i] = slack[i] - min_in_rows[l];  // subtract the minimum in row from that row

}