#include "includes.h"
__global__ void addPermutations(double *determinant, double *permutations, int *n){

int nn=*n**n-1;
*determinant=0;
for(int i=0;i<nn;i++){
*determinant+=permutations[i];
}
}