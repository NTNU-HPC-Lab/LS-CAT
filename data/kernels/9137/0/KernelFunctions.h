// KernelFunctions.h
// Contains wrapper calls around GPU code

#include <vector>

/**
 * Runs BFS using Thrust Vectors
 * graph       - contains all vertices and their edges
 * destination - destination vertex
 * source      - source vertex
 * totalEdges  - total number of edges in graph
 * 
**/
float RunBFSUsingThrust(std::vector<std::vector<int> > &graph,
                        int                             destination,
                        int                             source,
                        int                             totalEdges);


/**
 * Runs BFS on GPU by searching level by level
 * vertices    - list of vertices for GPU
 * edges       - list of edge destinations for GPU
 * vertIndices - list of start points for each vertices edges in edge list
 * edgeSize    - list of how many edges each vertex has
 * destination - destination vertex
 * source      - source vertex
 **/
float BFSByLevel(std::vector<int> &vertices,
                 std::vector<int> &edges,
                 std::vector<int> &vertIndices,
                 std::vector<int> &edgeLength,
                 int               destination,
                 int               source);
