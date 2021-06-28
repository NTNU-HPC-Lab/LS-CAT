#include "includes.h"
__global__ void k3(const int N, int* augPath, bool* visited, int* frontier, bool* new_frontier, bool* par_mat, int* cap_mat, bool* adj_mat, int* cap_max_mat, int* maxflow, bool* augFound) {
augFound[0] = false;

//Find the augmented path
augPath[0] = N - 1;
int i = 1, vertex = N - 1;
while(vertex != 0) {
for(int j = 0; j < N; j++) {
if(par_mat[vertex * N + j]) {
vertex = j;
augPath[i] = vertex;
i++;
break;
}
}
}

//Compute the bottleneck for the augmented path
int bottleneck = -1;
for(int i = 0; i < N; i++) {
if(augPath[i] == 0)
break;
else {
int k = augPath[i];
int j = augPath[i + 1];
int freeCap;
if(adj_mat[j * N + k]) {
freeCap = cap_max_mat[j * N + k] - cap_mat[j * N + k];
} else {
freeCap = cap_mat[k * N + j];
}

if(bottleneck == -1)
bottleneck = freeCap;
else if(freeCap < bottleneck)
bottleneck = freeCap;
}
}
maxflow[0] += bottleneck;

//Update capacities in d_cap_mat
for(int i = 0; i < N; i++) {
if(augPath[i] == 0)
break;
else {
int k = augPath[i];
int j = augPath[i + 1];
if(adj_mat[j * N + k]) {
cap_mat[j * N + k] += bottleneck;
} else {
cap_mat[k * N + j] -= bottleneck;
}
}
}

//Initialize par_mat
for(int i=0;i<N*N;i++)
par_mat[i] = false;

//Initialize visited and frontier
for(int i=0;i<N;i++) visited[i] = false;
for(int i=0;i<N;i++) new_frontier[i] = false;

visited[0] = true;
frontier[0] = 1;
frontier[1] = 0;
}