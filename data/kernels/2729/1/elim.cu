#include "includes.h"

using namespace std;
#define TILE 16


/* LU Decomposition using Shared Memory \
\           CUDA                        \
\										\
\ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/


//Initialize a 2D matrix
__global__ void elim(double *A, int n, int index, int bsize){
extern __shared__ double pivot[];

int idThread=threadIdx.x;
int idBlock=blockIdx.x;
int blockSize=bsize;


if(idThread==0){
for(int i=index;i<n;i++) pivot[i]=A[(index*n)+i];
}

__syncthreads();
//Varitables for pivot, row, start and end
int pivotRow=(index*n);
int currentRow=(((blockSize*idBlock) + idThread)*n);
int start=currentRow+index;
int end=currentRow+n;
//If greater than pivot row, loop from start index + 1(next row) to end of column
if(currentRow >pivotRow){
for(int i= start+1; i<end; ++i){
//Set the matrix value of next row and its column - pivot
A[i]=A[i]-(A[start]*pivot[i-currentRow]);

}
}
}