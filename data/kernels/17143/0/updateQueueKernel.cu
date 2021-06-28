#include "includes.h"
/*
Sample input file format:

1.Line : 6 => Number of nodes(int)
2.Line : 7 => Number of edges(int)
3.Line : 1 2 5.0 ----------------
4.Line : 2 3 1.5                |
5.Line : 1 3 2.1				|
6.Line : 1 4 1.2				|=> Edges
7.Line : 1 5 15.5				|
8.Line : 2 5 3.6				|
9.Line : 3 6 1.2-----------------
10.Line : 1 => Start node.
///////////////////////////////////////////////////////

Doesn't check any error condition.
*/


using namespace std;

// Edge struct.
typedef struct {
int* startPoints;
int* endPoints;
double* weights;
}Edge;


// This kernel will call queue size thread.

__global__ void updateQueueKernel(int *queueu,int *queueSize, const int *startPoints, const int *endPoints,const int*visitedArray, const int *currentVertex ) {

int index = threadIdx.x;
if (startPoints[index] == *currentVertex && visitedArray[endPoints[index]] == 0 )
{
int oldValue = atomicAdd(queueSize,1);
queueu[oldValue] = index;
}
}