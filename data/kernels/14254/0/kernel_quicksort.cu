#include "includes.h"








/*
* Naive sort
* used if the quicksort uses too many levels
*/
__global__ void kernel_quicksort(int* values, int n) {
#define MAX_LEVELS	1000

int pivot, L, R;
int idx =  threadIdx.x + blockIdx.x * blockDim.x;
int start[MAX_LEVELS];
int end[MAX_LEVELS];

start[idx] = idx;
end[idx] = n - 1;
while (idx >= 0) {
L = start[idx];
R = end[idx];
if (L < R) {
pivot = values[L];
while (L < R) {
while (values[R] >= pivot && L < R)
R--;
if(L < R)
values[L++] = values[R];
while (values[L] < pivot && L < R)
L++;
if (L < R)
values[R--] = values[L];
}
values[L] = pivot;

start[idx + 1] = L + 1;
end[idx + 1] = end[idx];
end[idx++] = L;


if (end[idx] - start[idx] > end[idx - 1] - start[idx - 1]) {
int tmp = start[idx];
start[idx] = start[idx - 1];
start[idx - 1] = tmp;

tmp = end[idx];
end[idx] = end[idx - 1];
end[idx - 1] = tmp;
}

}
else
idx--;
}
}