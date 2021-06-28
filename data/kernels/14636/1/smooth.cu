#include "includes.h"
__global__ void smooth( unsigned char *entrada,unsigned char *saida, int n_linhas, int n_colunas ) {
//Calcula a posição no vetor (id_bloco * total_blocos + id_thread)
int posicao = blockIdx.x * blockDim.x + threadIdx.x;
//Se a posição não é maior que o limite da imagem original...
if(posicao < (n_linhas)*(n_colunas)) {
//soma o valor da região 5x5 em torno no pixel
saida[posicao] =entrada[posicao]+
entrada[posicao+(n_colunas+4)]+
entrada[posicao+(2*(n_colunas+4))]+
entrada[posicao+(3*(n_colunas+4))]+
entrada[posicao+(4*(n_colunas+4))]+
entrada[posicao+1]+
entrada[posicao+(n_colunas+4)+1]+
entrada[posicao+(2*(n_colunas+4))+1]+
entrada[posicao+(3*(n_colunas+4))+1]+
entrada[posicao+(4*(n_colunas+4))+1]+
entrada[posicao+2]+
entrada[posicao+(n_colunas+4)+2]+
entrada[posicao+(2*(n_colunas+4))+2]+
entrada[posicao+(3*(n_colunas+4))+2]+
entrada[posicao+(4*(n_colunas+4))+2]+
entrada[posicao+3]+
entrada[posicao+(n_colunas+4)+3]+
entrada[posicao+(2*(n_colunas+4))+3]+
entrada[posicao+(3*(n_colunas+4))+3]+
entrada[posicao+(4*(n_colunas+4))+3]+
entrada[posicao+4]+
entrada[posicao+(n_colunas+4)+4]+
entrada[posicao+(2*(n_colunas+4))+4]+
entrada[posicao+(3*(n_colunas+4))+4]+
entrada[posicao+(4*(n_colunas+4))+4];
//calcula a média
saida[posicao] = saida[posicao]/25;
}
}