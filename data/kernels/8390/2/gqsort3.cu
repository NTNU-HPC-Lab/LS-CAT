#include "includes.h"
#define MAXR(sz) (((sz)+MAXSEQ-1)/MAXSEQ+1)
#define MAXT MAXR(MAXN)
int MAXN;
int MAXSEQ;
int THRN;

//===Definicion de estructuras y funciones utiles===

typedef struct secuence{
int start,end,pivot;
}secuence;

typedef struct block{
secuence seq,parent;
int blockcount,id,bid;
}block;

__global__ void gqsort3(block * blocks,int * d,int * _d){

int id = blockIdx.x,th = threadIdx.x,cth = blockDim.x;
int start = blocks[id].seq.start,end = blocks[id].seq.end;
for(int j=start+th;j<end;j+=cth)
d[j] = _d[j];

return;
}