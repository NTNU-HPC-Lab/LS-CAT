#include "includes.h"

float *A,*L,*U,*input;
void arrayInit(int n);
void verifyLU(int n);
void updateLU(int n);
void freemem(int n);

/*
*/


__global__ void scale( float *a, int b, int c) {
int index=c,size=b,k=0;

for(k=index+1;k<size;k++) {
a[size*index + k] = a[size*index + k] / a[size*index + index];
}

}