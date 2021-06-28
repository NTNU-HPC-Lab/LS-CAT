#include "includes.h"

using namespace std;

// this amazingly nice error checking function is stolen from:
//https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
__global__ void MatrixMulKernel(double *OutMat, double *Mat1, double *Mat2,  int Arows, int Acols, int Bcols) {
// row and column within submatrix
int blockrow =  blockIdx.y;//*
int row = threadIdx.y;
int blockcol = blockIdx.x;
int col =  threadIdx.x ;

// allocate these arrays only once we can change the values in them later
__shared__ double subAshared[BLOCKSIZE*BLOCKSIZE];
__shared__ double subBshared[BLOCKSIZE*BLOCKSIZE];
double Cvalue=0;

for (int B = 0; B < ceil((double)(Acols / BLOCKSIZE)) + 1; B++) {
// fetch from global memory
// yes, these took a LONG time to figure out. Pencil and Paper FTW!

/* notice:
1) how these indexes are actually offset a multiple of B, *not 1*.
2) threads are offset by col which will be 1 apart for each thread
3) which means that means all threads in the warp are hitting successive global memory cells
*/
int Mat1index = (row + blockrow*BLOCKSIZE)*Acols + col + B*BLOCKSIZE;
int Mat2index = (B*BLOCKSIZE + row)*Bcols + BLOCKSIZE*blockcol + col;

if (Mat1index < Arows*Acols)
subAshared[row*BLOCKSIZE + col] = Mat1[Mat1index];
else
subAshared[row*BLOCKSIZE + col] = 0;

if (Mat2index < Acols*Bcols)
subBshared[row*BLOCKSIZE + col] = Mat2[Mat2index];
else
subBshared[row*BLOCKSIZE + col] = 0;

__syncthreads();

// this computation is all using shared memory (fast)
for (int j = 0; j < BLOCKSIZE; j++)
if ((row*BLOCKSIZE + j < BLOCKSIZE*BLOCKSIZE) && (j*BLOCKSIZE + col < BLOCKSIZE*BLOCKSIZE))
Cvalue += subAshared[row*BLOCKSIZE + j]*subBshared[j*BLOCKSIZE + col];

__syncthreads();

}
if ((row < Arows) && (col < Bcols)) {
int finalmatrow = blockrow*BLOCKSIZE + row;
int finalmatcol = blockcol*BLOCKSIZE + col;
OutMat[finalmatrow*Bcols +  finalmatcol] = Cvalue;
}
}