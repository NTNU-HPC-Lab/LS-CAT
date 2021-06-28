#include "includes.h"
// Contains GPU Cuda code that executes BFS algorithm

// STL

// Internal Headers

// taken from global_memory.cu, Creates event and records time
__global__ void BFSLevels(int  *vertices, int  *edges, int  *distances, int  *predecessors, int  *vertIndices, int  *edgeSize, bool *levels, bool *visitedVertices, bool *foundDest, int   numVert, int   destination)
{
// Grab ThreadID
int thrID = threadIdx.x + blockIdx.x * blockDim.x;

__shared__ bool destFound;
destFound = false;

if (thrID < numVert && !destFound)
{
int curVert = vertices[thrID];

// Iterate through level if true
if (levels[curVert])
{
levels[curVert]          = false;
visitedVertices[curVert] = true;

// Grab indexes for curVert edges in edge array
int edgesEnd  = edgeSize[thrID] + vertIndices[thrID];

// Iterate through all edges for current vertex
for (int edgeIter = vertIndices[thrID]; edgeIter < edgesEnd; ++edgeIter)
{
// Grab next Vertex at end of edge
int nextVert = edges[edgeIter];

// If it hasn't been visited store info
// for distance and predecessors and set level
// to true for next level of vertices
if (!visitedVertices[nextVert])
{
distances[nextVert] = distances[curVert] + 1;
levels[nextVert] = true;
predecessors[nextVert]  = curVert;

// Set found destination to true and sync threads
if (nextVert == destination)
{
*foundDest = true;
destFound  = true;
__syncthreads();
}
}
}
}
}
}