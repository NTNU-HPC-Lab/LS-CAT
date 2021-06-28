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
__global__ void Update_Paths(Vertex *vertices, int *length, int *updateLength)
{
int u = threadIdx.x;
if(length[u] > updateLength[u])
{

length[u] = updateLength[u];
vertices[u].visited = FALSE;
}

updateLength[u] = length[u];


}