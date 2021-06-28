#include "includes.h"

#define VERTICES 600

__constant__ float2 d_vertices[VERTICES];
__constant__ float d_slopes[VERTICES];

/*
* This file contains the implementation of a CUDA Kernel for the
* point-in-polygon problem using the crossing number algorithm
*
* The kernel cn_pnpoly is can be tuned using the following parameters:
*    * block_size_x                any sensible thread block size
*    * tile_size                   any sensible tile size value
*    * between_method              any of [0, 1, 2, 3]
*    * use_precomputed_slopes      enable or disable [0, 1]
*    * use_method                  any of [0, 1]
*
* The kernel cn_pnpoly_naive is used for correctness checking.
*
* The algorithm used here is adapted from:
*     'Inclusion of a Point in a Polygon', Dan Sunday, 2001
*     (http://geomalgorithms.com/a03-_inclusion.html)
*
* Author: Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
*/

#ifndef block_size_x
#define block_size_x 256
#endif
#ifndef block_size_y
#define block_size_y 1
#endif
#ifndef block_size_z
#define block_size_z 1
#endif

#ifndef tile_size
#define tile_size 1
#endif



__global__ void cn_pnpoly_naive(int* bitmap, float2* points, int n) {
int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i < n) {
int c = 0;
float2 p = points[i];

int k = VERTICES-1;

for (int j=0; j<VERTICES; k = j++) {    // edge from v to vp
float2 vj = d_vertices[j];
float2 vk = d_vertices[k];

float slope = (vk.x-vj.x) / (vk.y-vj.y);

if ( (  (vj.y>p.y) != (vk.y>p.y)) &&            //if p is between vj and vk vertically
(p.x < slope * (p.y-vj.y) + vj.x) ) {   //if p.x crosses the line vj-vk when moved in positive x-direction
c = !c;
}
}

bitmap[i] = c; // 0 if even (out), and 1 if odd (in)
}


}