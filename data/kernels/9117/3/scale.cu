#include "includes.h"
__global__ void scale( float *a, int size, int c) {
int index=c,k=0;//size=b

for(k=index+1;k<size;k++) {
a[size*index + k] = (float) a[size*index + k] / a[size*index + index];
}

}