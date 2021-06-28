#include "includes.h"


__global__ void calculoAlgoritmoTroca(float *dev_matrizSuperior, int linhaPerm, int colunaPerm, int totalColunas, int totalLinhas)
{
int i = blockDim.x * blockIdx.x + threadIdx.x;
float fatorAnulador = 0.0;

//evitar operação em endereço invalido
//se for indice da linha permissivel, desconsiderar
if (i > totalLinhas || i == linhaPerm)
return;

//computar fator anulador da respectiva linha
fatorAnulador = dev_matrizSuperior[i * totalColunas + colunaPerm] * (-1);

//calcular os valores dos elementos da linha usando o fator anulador coletado
for (int coluna = 0; coluna < totalColunas; coluna++){

if (i * totalColunas + coluna > totalLinhas * totalColunas)
return;

//o valor da coluna permissivel sera 0
if (coluna == colunaPerm)
dev_matrizSuperior[i * totalColunas + coluna] = 0;
else
//os demais valores devem respeitar a equacao
//Valor = FatorAnulador * ValorRefLinhaPerm + LinhaAtual;
dev_matrizSuperior[i * totalColunas + coluna] = fatorAnulador
* dev_matrizSuperior[linhaPerm * totalColunas + coluna]
+ dev_matrizSuperior[i * totalColunas + coluna];
}

}