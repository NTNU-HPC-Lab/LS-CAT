#include "includes.h"
__global__ void kernel_div(char* newB, char* first, char* second, int size_first, int size_second, int * size_newB, char* aux) {
int i = threadIdx.x;
int j = threadIdx.y;

if(j==0 && i==0){
if(first[j]=='-' || second[i]=='-')
newB[0]='-';
else
newB[0]='+';
return;
}

#if __CUDA_ARCH__>=200
printf("#i, j = %d, %d\n", i, j);
#endif
// adapted from kernel_sub
int diff = size_first - size_second;
int tmp = 0;
if (j - 1 - diff >= 0 && (second[j - 1 - diff] != '+' && second[j - 1 - diff] != '-')) {
tmp = first[j - 1] - second[j-1-diff];
} else if (first[j - 1] != '+' && first[j - 1] != '-') {
tmp = first[j - 1];
}

if (tmp < 0) {
// warning 10 - tmp ?
aux[i * size_first + j - 1]--;
tmp += 10;
}
if (i != 0)
aux[i * size_first + j] += tmp;
// end of kernel_sub

#if __CUDA_ARCH__>=200
printf("#aux = %d\n", aux[i * size_first + j]);
#endif

/*
char* temp = NULL;
//init(size_second + 1, temp);
int t = 0; // temp's index
int n = 0; // newB's index
for (int i = size_first - 1; i >= 0; i -= t) {
t = 0;
for (int j = i - size_second; j <= i; j++) {
if (j >= 0) {
temp[t] = first[j];
t++;
}
}
// verify that we are not attempting to divide something too small
if (isFirstBiggerThanSecond(second, temp, size_second)) {
t = 0;
for (int j = i - size_second - 1; j <= i; j++) {
if (j < 0) {
// nothing left to divide, exit function
return;
} else {
temp[t] = first[j];
t++;
}
}
}
// now that we have our thing, let's get to the division itself
char res = 0;
char* sub_res = NULL;
int size_res = 0;
//init(size_second, sub_res);
do {
//kernel_sub(sub_res, temp, second, size_second, size_second, &size_res);
res++;
} while (0); //sub_res > 0
// current division done, save result & move on to the next
newB[n] = res;
n++;
}
// all divisions done, we need to realign our result;
int diff = size_second - n;
for (int i = size_second - 1; i > n; i++) {
newB[i] = newB[i - diff];
}*/
}