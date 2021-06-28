#include "includes.h"
__global__ void glcm_calculation(int *A,int *glcm,float *glcmNorm, const int nx, const int ny,int maxx)
{

int ix = threadIdx.x + blockIdx.x * blockDim.x;
int iy = threadIdx.y + blockIdx.y * blockDim.y;
unsigned int idx = iy * nx + ix;



//unsigned int idr = iy * (maxx+1) + ix;


int k,l;
int p;


//Calculate GLCM
if(idx < nx*ny ){
for(k=0;k<=maxx;k++){
for(l=0;l<=maxx;l++){
if((A[idx]==k) && (A[idx+1]==l)){
p=((maxx+1)*k) +l;
glcm[p]+=1;
}
}
}
}


//Normalization
int sum;
sum = 0;
if(idx<(maxx+1)*(maxx+1)){
for(k=0;k<((maxx+1)*(maxx+1));k++){
sum+=glcm[k];
}
}
// if(ix<1){
//     printf("sum %d \n ",sum);
// }
if(idx<((maxx+1)*(maxx+1))){
glcmNorm[idx] = float(glcm[idx])/float(sum);
}

float sums;

if(ix<1){
for(k=0;k<((maxx+1)*(maxx+1));k++){
sums += glcmNorm[k];

}
}

float f1;

f1=0;
if(ix<1){
for(k=0;k<((maxx+1)*(maxx+1));k++){
f1 = f1 + glcmNorm[k];

}
}
//mat[offset] = sqrt(mat[offset]);

float f2 = 0;
if(ix<1){
for(k=0;k<((maxx+1)*(maxx+1));k++){
f2 = f2 + k*k*sums;

}
}

float f3;
f3 = sqrt(f1);


float f4;

if(ix<1){
for(k=0;k<((maxx+1)*(maxx+1));k++){
f4 += (glcmNorm[k] * log10f(glcmNorm[k]));

}
}

//float sum_average=0;


// float f5;
// if(ix<1){
//     for(k=0;k<((maxx+1)*(maxx+1));k++){
//         f2 = f2 + k*k*sums;

//     }
// }

// for (int j = 0, int i = 0; j<DIM, i<DIM; j++,i++){
//         for (int k = DIM*j; k<DIM*(j+1); k++)
//         f5 += i*mat[k];
//     }

// float f6;
// for (int i = 0; i<DIM; i++ ){
//     mat2[offset]= (i-f5)*(i-f5)*mat[offset];
//     for (int j=0; j<DIM; j++){
//         f6 += mat2[row*DIM*j];
//     }
// }
// if(row<DIM){
//         printf("array di device %d : %f \n",offset,mat[tidx]);
//         //printf("array di device %d : %f \n",offset,mat2[tidx]);
//         //mat[offset]=mat[offset]/sum;
// }
if(ix<1){
printf("ASM : %.1f\n", f1);
printf("Contrast : %.1f\n",f2);
printf("Energy : %.1f\n",f3);
printf("Entropy : %.1f\n",f4);
//printf("Miu : %.1f\n",f5);
//printf("Variance : %.1f\n",f6);
}

}