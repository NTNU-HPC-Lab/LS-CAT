#include "includes.h"
//  Author: Jose F. Martinez Rivera
//  Course: ICOM4036 - 040
//	Professor: Wilson Rivera Gallego
// 	Assignment 2 - CUDA Implementation



#define V 8
#define E 11
#define MAX_WEIGHT 1000000
#define TRUE    1
#define FALSE   0

typedef int boolean;
//
//Represents an edge or path between Vertices
typedef struct
{
int u;
int v;

} Edge;

//Represents a Vertex
typedef struct
{
int title;
boolean visited;

} Vertex;


//Finds the weight of the path from vertex u to vertex v
__device__ __host__ int findEdge(Vertex u, Vertex v, Edge *edges, int *weights)
{

int i;
for(i = 0; i < E; i++)
{

if(edges[i].u == u.title && edges[i].v == v.title)
{
return weights[i];
}
}

return MAX_WEIGHT;

}
__global__ void Find_Vertex(Vertex *vertices, Edge *edges, int *weights, int *length, int *updateLength)
{

int u = threadIdx.x;


if(vertices[u].visited == FALSE)
{


vertices[u].visited = TRUE;


int v;

for(v = 0; v < V; v++)
{
//Find the weight of the edge
int weight = findEdge(vertices[u], vertices[v], edges, weights);

//Checks if the weight is a candidate
if(weight < MAX_WEIGHT)
{
//If the weight is shorter than the current weight, replace it
if(updateLength[v] > length[u] + weight)
{
updateLength[v] = length[u] + weight;
}
}
}

}

}