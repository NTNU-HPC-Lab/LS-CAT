#include "includes.h"

using namespace std;
#define TILE 16


/* LU Decomposition using Shared Memory \
\           CUDA                        \
\										\
\ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/


//Initialize a 2D matrix
__global__ void scaleIndex(double *matrix, int n, int index){
int start=(index*n+index);
int end=(index*n+n);

for(int i= start+1 ; i<end; ++i){
matrix[i]=(matrix[i]/matrix[start]);
}

}