#include "includes.h"
__global__ void splitNodes(int* octree, int* numNodes, int poolSize, int startNode) {

int index = blockIdx.x * blockDim.x + threadIdx.x;

//Don't do anything if its out of bounds
if (index < poolSize) {
int node = octree[2 * (index+startNode)];

//Split the node if its flagged
if (node & 0x40000000) {
//Get a new node tile
int newNode = atomicAdd(numNodes, 8);

//Point this node at the new tile
octree[2 * (index+startNode)] = (octree[2 * (index+startNode)] & 0xC0000000) | (newNode & 0x3FFFFFFF);

//Initialize new child nodes to 0's
for (int off = 0; off < 8; off++) {
octree[2 * (newNode + off)] = 0;
octree[2 * (newNode + off) + 1] = 0;
}
}
}

}