#include "includes.h"
__global__ void multiplyDet(double *matrix, double *determinant,int* n){

int i;
int nn=*n;
for(i=0;i<nn;i++){
*determinant=(*determinant)*matrix[i*(nn)+i];
}
}