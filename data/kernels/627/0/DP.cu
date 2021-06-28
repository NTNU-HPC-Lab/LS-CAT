#include "includes.h"



// Lenght of each data
__constant__ int gcT_size;
__constant__ int gcP_size;

// Threshold of the SW algorithm
__constant__ int gcThre;

// Data of the query
__constant__ char gcP_seq[1024];

// Cost and Gain
__constant__ int gcMatch;
__constant__ int gcMiss;
__constant__ int gcExtend;
__constant__ int gcBegin;

enum{
Zero,
Diagonal,
Vertical,
Horizon,
};

using namespace std;

__global__ void DP(char* dT_seq, char* dTrace, int* dScore){
// ThreadId = ptn point
int id = threadIdx.x;
// The acid in this thread
char p = gcP_seq[id];
// p-1 row line's value
__shared__ int Hp_1[1024];
__shared__ int Ep_1[1024];
// Temporary
int Hp_1_buf = 0;
int Ep_1_buf = 0;
// t-1 element value
int Ht_1 = 0;
int Ft_1 = 0;
// p-1 t-1 element value
int Ht_1p_1 = 0;
// Initialize
Hp_1[id] = 0;
Ep_1[id] = 0;
// Similar score
int sim = 0;
int point = id * gcT_size - id;
// Culcurate elements
for(int t = -id; t < gcT_size; ++t){
// Control culcurate order
if(t<0){}
// Get similar score
else{
// Compare acids
if(dT_seq[t] == p){sim = gcMatch;}
else{sim = gcMiss;}
}
// SW algorithm
// Culcurate each elements
Ht_1p_1 += sim;	// Diagonal
Ht_1 += gcBegin;	// Horizon (Start)
Ft_1 += gcExtend;	// Horizon (Extend)
Hp_1_buf = Hp_1[id] + gcBegin;	// Vertical (Start)
Ep_1_buf = Ep_1[id] + gcExtend;	// Vertical (Extend)
// Choose the gap score
if(Ht_1 > Ft_1){Ft_1 = Ht_1;}	// Horizon
if(Hp_1_buf > Ft_1){Ep_1_buf = Hp_1_buf;}	// Vertical
// Choose the max score
// Ht_1 is stored the max score
if(Ht_1p_1 > Ep_1_buf){
// Diagonal
if(Ht_1p_1 > Ft_1){
Ht_1 = Ht_1p_1;
dTrace[point] = Diagonal;
}
// Horizon
else{
Ht_1 = Ft_1;
dTrace[point] = Horizon;
}
}
else {
// Vertical
if(Ep_1_buf > Ft_1){
Ht_1 = Ep_1_buf;
dTrace[point] = Vertical;
}
// Horizon
else{
Ht_1 = Ft_1;
dTrace[point] = Horizon;
}
}
// The case 0 is max
if(Ht_1 <= 0){
Ht_1 = 0;
// Set 0 other value
Ft_1 = 0;
Ep_1_buf = 0;
dTrace[point] = Zero;
}
// Hp-1 is next Ht-1p-1
Ht_1p_1 = Hp_1[id];
__syncthreads();
// Set value need next culcurate
// p+1 row line
if(t >= 0){
Hp_1[id + 1] = Ht_1;
Ep_1[id + 1] = Ep_1_buf;
// DEBUG, score check
// dTrace[point] = (char)(Ht_1);
}
if(Ht_1 >= gcThre){
//		printf("Score = %d:\n", Ht_1);
// traceback(dTrace, dT_seq, point-1, t);
if(Ht_1 >= (dScore[t] & 0x0000ffff)){
// Set score and now ptn point
dScore[t] = Ht_1 + (id << 16);
}
}
++point;
__syncthreads();
// for end
}
}