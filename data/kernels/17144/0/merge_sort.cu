#include "includes.h"


using namespace std;

// 用宏变长参数来实现
__global__ void merge_sort(int *datas, int n){
int tid=blockDim.x*threadIdx.y+threadIdx.x;
extern __shared__ int shared[];
if (tid<n) shared[tid] = datas[tid];
__syncthreads();
int cnt=1;
for (int gap=2; gap<n*2; gap<<=1, cnt++){
if (tid%gap==0){
int left=tid+n*((cnt+1)%2);
int mid=tid+gap/2+n*((cnt+1)%2);
int right=mid;
int end=tid+gap+((cnt+1)%2)*n;
int full_end=(1+(cnt+1)%2)*n;
int res_ind=n*(cnt%2)+tid;

while((left<mid && left<full_end) || (right<end && right<full_end)){
if (!(left<mid && left<full_end)){
shared[res_ind]=shared[right];
right++;
}else if (!(right<end && right<full_end)){
shared[res_ind]=shared[left];
left++;
}else{
if (shared[right]> shared[left]){
shared[res_ind]=shared[left];
left++;
}else{
shared[res_ind]=shared[right];
right++;
}
}
res_ind++;
}
}
__syncthreads();
}

datas[tid]=shared[tid+ ((cnt+1)%2)*n];
}