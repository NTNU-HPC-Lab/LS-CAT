#include "includes.h"
__global__ void block_QR(float* z, float* z1, float* vector, float* vector1, float* Q, float* NewQ, float* R, float* PrevM, float* NewM, int* converged, float* eigenvector, const int *WidthOfMatrix, const int *ind, const int *vind)
{
//extern __shared__ float z1[];
int n = WidthOfMatrix[blockIdx.x];
int index = ind[blockIdx.x];
int vectindex = vind[blockIdx.x];
int numofelements = n*n;
if(threadIdx.x==0){
converged[blockIdx.x] = 0;
}
if(threadIdx.x<numofelements){
int i;
for(i=threadIdx.x;i<numofelements;i=i+blockDim.x){
//set eigenvector to the identity matrix.
if(i/n==i%n)eigenvector[i+index]=1;
else eigenvector[i+index]=0;
}
for(i=threadIdx.x;i<numofelements;i=i+blockDim.x){
int iplusindex = i+index;
z1[iplusindex]=z[iplusindex];
Q[iplusindex]=z[iplusindex];
PrevM[iplusindex]=z[iplusindex];
}
do{
int k, j, PowOf2;
for(k=0;k<n-1;k++){
//Householder Code
//STEP 0: Get value of z[k*n+k] for use in step 4
float NormCheck = z[k*n+k+index];
//STEP 1: Find minor matrix of the input matrix z and sets it to z
for(i=threadIdx.x;i<numofelements;i=i+blockDim.x){
if(i%n==i/n&&i/n<k)z[i+index]=1;
else if(i/n>=k&&i%n>=k)z[i+index]=z[i+index];
else z[i+index]=0;
}
__syncthreads();
//STEP 2: Find kTH column of z and set to vector
for(i=threadIdx.x;i<numofelements;i=i+blockDim.x){
if(i<n){
vector[i+vectindex] = z[i*n+k+index];
}
}
//STEP 3: Find the norm of the kTh column and set to NormOfKcol
float NormOfKcol;
for(i=threadIdx.x;i<numofelements;i=i+blockDim.x){
if(i<n){
int iplusvectindex = i + vectindex;
//Need a temporary for vector that we can change the values of since we need mcol later
vector1[iplusvectindex] = vector[iplusvectindex];
vector1[iplusvectindex] *= vector1[iplusvectindex];
}
}
PowOf2 = 1;
__syncthreads();
//add all x's together, 2 at a time. O((log(n)) function
for(i = 0;i < ((float)n)/2.0;i++){
for(j=threadIdx.x;j<numofelements;j=j+blockDim.x){
if(j<n&&j%(PowOf2*2)==0&&j+PowOf2<n){
int jplusvectindex = j + vectindex;
vector1[jplusvectindex] = vector1[jplusvectindex] + vector1[PowOf2+jplusvectindex];
}
}
__syncthreads();
//PowOf2 = pow(2,i)
for(j=threadIdx.x;j<numofelements;j=j+blockDim.x){
if(j<n){
PowOf2 *= 2;
}
}
}
NormOfKcol = sqrt(vector1[0+vectindex]);
//STEP 4: Make Norm Negative if NormCheck is > 0
if(NormCheck > 0) NormOfKcol = -NormOfKcol;
//STEPS 5+6 Combined: add NormOfKcol to tmp[k]
if(k==threadIdx.x)vector[k+vectindex]=vector[k+vectindex]+NormOfKcol;
__syncthreads();
//STEP 7: Finds the addition of the new kcol and stores it in tmp[0]
//used in ||tmp||
for(i=threadIdx.x;i<numofelements;i=i+blockDim.x){
if(i<n){
int iplusvectindex = i + vectindex;
vector1[iplusvectindex] = vector[iplusvectindex] * vector[iplusvectindex];
PowOf2 = 1;
}
}
__syncthreads();
//add all tmp's together, 2 at a time. O(n(log(n)) function
for(i = 0;i < ((float)n)/2.0;i++){
for(j=threadIdx.x;j<numofelements;j=j+blockDim.x){
if(j<n&&j%(PowOf2*2)==0&&j+PowOf2<n){
int jplusvectindex = j + vectindex;
vector1[jplusvectindex] = vector1[jplusvectindex] + vector1[PowOf2+jplusvectindex];
}
}
__syncthreads();
//PowOf2 = pow(2,i)
for(j=threadIdx.x;j<numofelements;j=j+blockDim.x){
if(j<n){
PowOf2 *= 2;
}
}
}
__syncthreads();
//STEP 8: Divide vector Vmadd by the Norm[0] and set it to Vdiv
// Vdiv = Vmadd / norm
for(i=threadIdx.x;i<numofelements;i=i+blockDim.x){
if(i<n){
int iplusvectindex = i + vectindex;
vector[iplusvectindex] = vector[iplusvectindex]/(sqrt(vector1[vectindex]));
}
}
__syncthreads();
//STEP 9: Multiply the Vdiv vector by its transverse and subtract that from I, store the resulting matrix in Vmul
// Vmul = I - 2 * Vdiv * Vdiv^T
//threadIdx.x%n = column
//threadIdx.x/n = row (integer division)
for(i=threadIdx.x;i<numofelements;i=i+blockDim.x){
R[i+index] = -2 * vector[i/n+vectindex] * vector[i%n+vectindex];
}
//if on the diagonal(row==column)
for(i=threadIdx.x;i<numofelements;i=i+blockDim.x){
if(i/n==i%n){
R[i+index] += 1;
}
}
__syncthreads();
//STEP 10: Multiply Vmul by input matrix z1 and store in VmulZ
for(i=threadIdx.x;i<numofelements;i=i+blockDim.x){
z[i+index]=0;
for(j=0;j<n;j++){
z[i+index]+= R[i/n*n+j+index] * z1[j*n+i%n+index];
}
}
//STEP 11: if k!=0 Multiply Vmul by input matrix Q and set to NewQ
if(k!=0){
for(i=threadIdx.x;i<numofelements;i=i+blockDim.x){
NewQ[i+index]=0;
for(j=0;j<n;j++)
{
NewQ[i+index]+= R[i/n*n+j+index] * Q[j*n+i%n+index];
}
}
}
__syncthreads();
//STEP 12.1: If first iteration of k, set Q to vmul for use in next iteration of k
if(k==0){
for(i=threadIdx.x;i<numofelements;i=i+blockDim.x){
Q[i+index] = R[i+index];
}
}
//STEP 12.2: If after first iteration of k, set Q to NewQ, which was found by multiplying the old Q by Vmul.
else {
for(i=threadIdx.x;i<numofelements;i=i+blockDim.x){
Q[i+index] = NewQ[i+index];
}
}
//STEP 12.3: Set z and z1 to VmulZ for use in the next iteration of k.
for(i=threadIdx.x;i<numofelements;i=i+blockDim.x){
z1[i+index] = z[i+index];
}
__syncthreads();
}
//Once for loop is completed:
//STEP 13: Multiply matrices Q and m to find the matrix R
for(i=threadIdx.x;i<numofelements;i=i+blockDim.x){
R[i+index]=0;
}
for(i=0;i<n;i++)
{
for(j=threadIdx.x;j<numofelements;j=j+blockDim.x){
R[j+index]+= Q[j/n*n+i+index] * PrevM[i*n+j%n+index];
}
}
__syncthreads();
//STEP 14: Find the transpose of matrix Q and store int TransposeOfQ
//threadIdx.x%n = column
//threadIdx.x/n = row (integer division)
// # -> #%n*n+#/n
// for n=4		0->0  1->4  2->8   3->12
//			4->1  5->5  6->9   7->13
//			8->2  9->6  10->10 11->14
//			12->3 13->7 14->11 15->15
for(i=threadIdx.x;i<numofelements;i=i+blockDim.x){
z[i%n*n+i/n+index] = Q[i+index];
}
__syncthreads();
//STEP 14.5: Multiply matrices eigenvector and TransposeOfQ and store in eigenvector(use NewM as a temporary matrix)
//NewM contains new eigenvectors
for(i=threadIdx.x;i<numofelements;i=i+blockDim.x){
NewM[i+index]=0;
for(j=0;j<n;j++){
NewM[i+index]+= eigenvector[i/n*n+j+index] * z[j*n+i%n+index];
}
}
__syncthreads();
for(i=threadIdx.x;i<numofelements;i=i+blockDim.x){
eigenvector[i+index]=NewM[i+index];
}
__syncthreads();
//STEP 15: Multiply matrices R and TransposeOfQ and store in NewM matrix
for(i=threadIdx.x;i<numofelements;i=i+blockDim.x){
NewM[i+index]=0;
for(j=0;j<n;j++)
{
NewM[i+index]+= R[i/n*n+j+index] * z[j*n+i%n+index];
}
}
//STEP 16: Check for Convergence of New Matrix (Newm)
if(threadIdx.x==0){
converged[blockIdx.x] = 1;
}
__syncthreads();
//threadIdx.x%n = column
//threadIdx.x/n = row (integer division)
for(i=threadIdx.x;i<numofelements;i=i+blockDim.x){
if(i/n==i%n&&(PrevM[i+index]/NewM[i+index]>1.000001||
PrevM[i+index]/NewM[i+index]<0.999999)){
converged[blockIdx.x] = 0;
}
}
__syncthreads();
//STEP 17: Set up for next iteration if converged is 0
if(converged[blockIdx.x]==0){
for(i=threadIdx.x;i<numofelements;i=i+blockDim.x){
int iplusindex = i + index;
z[iplusindex] = NewM[iplusindex];
z1[iplusindex] = NewM[iplusindex];
Q[iplusindex] = NewM[iplusindex];
PrevM[iplusindex] = NewM[iplusindex];
}
}
__syncthreads();
}while(converged[blockIdx.x]==0);
//put eigenvalues into vector
if(threadIdx.x<n){
vector[threadIdx.x+vectindex]=NewM[threadIdx.x+threadIdx.x*n+index];
}
__syncthreads();
if(threadIdx.x==0){
//Sort Eigenvalues low to high and swap eigenvectors to match eigenvalues
//Simple Bubble Sort
int i1,i2,i3;
for(i1=vectindex;i1<n-1+vectindex;i1++){
for(i2=i1+1;i2<n+vectindex;i2++){
if(vector[i1]>vector[i2]){
float tmp = vector[i1];
vector[i1] = vector[i2];
vector[i2] = tmp;
for(i3 = 0;i3<n;i3++){
float tmp = eigenvector[i3*n+(i1-vectindex)%n+index];
eigenvector[i3*n+(i1-vectindex)%n+index] = eigenvector[i3*n+(i2-vectindex)%n+index];
eigenvector[i3*n+(i2-vectindex)%n+index] = tmp;
}
}
}
}
}

}
}