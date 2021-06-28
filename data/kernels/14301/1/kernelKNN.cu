#include "includes.h"
__device__ float maxMetricPoints(const float* g_uquery, const float* g_vpoint, int pointdim, int signallength){
float	r_u1;
float	r_v1;
float	r_d1,r_dim=0;

r_dim=0;
for(int d=0; d<pointdim; d++){
r_u1 = *(g_uquery+d*signallength);
r_v1 = *(g_vpoint+d*signallength);
r_d1 = r_v1 - r_u1;
r_d1 = r_d1 < 0? -r_d1: r_d1;  //abs
r_dim= r_dim < r_d1? r_d1: r_dim;
}
return r_dim;
}
__device__ float insertPointKlist(int kth, float distance, int indexv,float* kdistances, int* kindexes){
int k=0;
while( (distance>*(kdistances+k)) && (k<kth-1)){k++;}
//Move value to the next
for(int k2=kth-1;k2>k;k2--){
*(kdistances+k2)=*(kdistances+k2-1);
*(kindexes+k2)=*(kindexes+k2-1);
}
//Replace
*(kdistances+k)=distance;
*(kindexes+k)=indexv;

//printf("\n -> Modificacion pila: %.f %.f. New max distance: %.f", *kdistances, *(kdistances+1), *(kdistances+kth-1));
return *(kdistances+kth-1);
}
__global__ void kernelKNN(const float* g_uquery, const float* g_vpointset, int *g_indexes, float* g_distances, int pointdim, int triallength, int signallength, int kth, int exclude)
{

const unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;
//const unsigned int tidim = tid*pointdim;
const unsigned int itrial = tid / triallength;  //  indextrial

int kindexes[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
float kdistances[]= {INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, \
INFINITY, INFINITY, INFINITY, INFINITY, INFINITY};

if(tid<signallength){

//int   r_index;
float r_kdist=INFINITY;
int indexi = tid-triallength*itrial;
for(int t=0; t<triallength; t++){
int indexu = tid;
int indexv = (t + itrial*triallength);
int condition1=indexi-exclude;
int condition2=indexi+exclude;
if((t<condition1)||(t>condition2)){
float temp_dist = maxMetricPoints(g_uquery+indexu, g_vpointset+indexv,pointdim, signallength);
if(temp_dist <= r_kdist){
r_kdist = insertPointKlist(kth,temp_dist,t,kdistances,kindexes);
//printf("\nId: %d, Temp_dist: %.f. r_index: %d", tid, temp_dist, r_index);
}
}
//printf("tid:%d indexes: %d, %d distances: %.f %.f\n",tid, *kindexes, *(kindexes+1), *kdistances, *(kdistances+1));
}

__syncthreads();
//COPY TO GLOBAL MEMORY
for(int k=0;k<kth;k++){
g_indexes[tid+k*signallength] = *(kindexes+k);
g_distances[tid+k*signallength]= *(kdistances+k);
}

}

}