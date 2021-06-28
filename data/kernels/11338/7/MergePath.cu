#include "includes.h"
__device__ int satisfies(int i, int j, int *A, int *B)
{
return (A[i] <= B[j]);
}
__global__ void MergePath(int *A, int *B, int* C, int *x, int *y, int n)
{
int num_of_threads = blockDim.x;
int idx = threadIdx.x;
bool flag = false;
if (idx == 0)
{
x[idx] = 0;
y[idx] = 0;
flag = true;
}
int A_start = idx*(2 * n) / num_of_threads; //only when len(A)==len(B)
int B_start = max(0, A_start - (n - 1));
A_start = min(n - 1, A_start);
int length_of_array;

if (B_start == 0)
{

length_of_array = A_start + 1;
}
else
length_of_array = n - B_start;

int left = 0, right = length_of_array - 1;
// cout<<A_start<<" "<<B_start<<" "<<length_of_array<<endl<<"-------------------------------------------\n";

while (left <= right && !flag)
{
// cout<<left<<" "<<right<<endl;
int mid = left + (right - left) / 2;
int I = A_start - mid;
int J = B_start + mid;
if (!satisfies(I, J, A, B))
{
left = mid + 1;
}
else
{
if (J == 0)
{
x[idx] = (I + 1);
y[idx] = (J);
flag = true;
}
else if (I == n - 1)
{
x[idx] = (I + 1);
y[idx] = (J);
flag = true;
}
else
{
if (!satisfies(I + 1, J - 1, A, B))
{
x[idx] = (I + 1);
y[idx] = (J);
flag = true;
}
else
{
right = mid;
}
}
}
}
left--;
if (!flag)
{
x[idx] = (A_start - left);
y[idx] = (n);
}
__syncthreads();

int end_x, end_y;
if (idx == num_of_threads - 1)
{
end_x = n;
end_y = n;
}
else
{
end_x = x[idx + 1];
end_y = y[idx + 1];
}
int cur_x = x[idx];
int cur_y = y[idx];
int put_at = cur_x + cur_y;
while (cur_x<end_x && cur_y<end_y)
{
if (A[cur_x] <= B[cur_y])
{
C[put_at++] = A[cur_x++];
}
else
{
C[put_at++] = B[cur_y++];
}
}
while (cur_x<end_x)
C[put_at++] = A[cur_x++];
while (cur_y<end_y)
C[put_at++] = B[cur_y++];
}