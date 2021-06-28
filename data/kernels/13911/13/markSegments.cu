#include "includes.h"
__global__ void markSegments(	unsigned short * d_mark, unsigned int 	circuitGraphEdgeCount, unsigned int * 	d_cg_edge_start, unsigned int *	d_cedgeCount, unsigned int 	circuitVertexSize){

unsigned int tid=(blockDim.x*blockDim.y * gridDim.x*blockIdx.y) + (blockDim.x*blockDim.y*blockIdx.x)+(blockDim.x*threadIdx.y)+threadIdx.x;
if(tid<circuitVertexSize){
d_mark[ d_cg_edge_start[tid]]=d_cedgeCount[tid];
}

}