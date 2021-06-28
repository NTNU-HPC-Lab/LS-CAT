#include "includes.h"
using namespace std;


__device__ void swap(int *a, int *b) {
int temp = *a;
*a = *b;
*b = temp;
}
__global__ void sort(int *d_arr, int n, bool isEven) {
int i;
if (isEven) {
i = threadIdx.x * 2;
} else {
i = threadIdx.x * 2 + 1;
}

if (i < n -1) {
if (d_arr[i] > d_arr[i + 1]) {
swap(&d_arr[i], &d_arr[i + 1]);
}
}
}