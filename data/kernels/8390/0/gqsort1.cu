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

__global__ void gqsort1(block * blocks,int * d,int * LT,int * GT){

int id = blockIdx.x,th = threadIdx.x,cth = blockDim.x;
int gt=0,lt=0,pivot=blocks[id].seq.pivot;
int start = blocks[id].seq.start,end = blocks[id].seq.end;

if(th==0){
LT[id]=0;
GT[id]=0;
}
__syncthreads();

for(int j=start+th;j<end;j+=cth){
if(d[j]<pivot)lt++;
else if(d[j]>pivot)gt++;
}

atomicAdd(&LT[id],lt);
atomicAdd(&GT[id],gt);

return;
}