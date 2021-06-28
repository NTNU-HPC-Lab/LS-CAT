#include "includes.h"
/****************************************************************************
Floyd - Warshall Algorithm developed using CUDA. A 2011-2012 assignement for
Parallel Programming Course of Electrical and Computer Engineering Department
in the Aristotle Faculty of Enginnering - Thessaloniki.

*****************************************************************************/


#define INF 100000000
#define BLOCKSIZE 128
#define BITSFT 7 //log2(BLOCKSIZE)


/*****************************************
Array Generator - filling weight matrices
according to Floyd-Warshall theory.
******************************************/
__global__ void pFloyd(float *D,float *Q,int vertices,int k,int k2)
{
int i,j,index;
i= blockIdx.x;
j=(blockIdx.y << BITSFT) + threadIdx.x;
index=(i << vertices)+j; 				//vertices equals log2(vertices).
if((D[(i << vertices)+k]+D[(k2)+j])<D[index])
{
D[index]=D[(i << vertices)+k]+D[(k2)+j];
Q[index]=k;
}
}