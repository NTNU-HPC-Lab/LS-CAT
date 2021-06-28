#include "includes.h"

#define MINVAL 1e-7

__global__ void MaxElement(double* Mtr, int Size, int i, int*strnum)
{
double MaxValue=Mtr[i*Size+i];
*strnum=i;

for(int k=i; k<Size; k++)
{

if(fabs(Mtr[i*(Size)+k])>fabs(MaxValue))
{
*strnum=*strnum+1;    //ýòî äëÿ êîìïèëÿòîðà ÷åêåðà
*strnum=k;
MaxValue=Mtr[i*(Size)+k];
}
}

if(fabs(MaxValue)<MINVAL)   //åñëè ìàêñèìàëüíûé ýëåìåíò íèæå ïîðîãîâîãî çíà÷åíèÿ, òî âîçâðàùàåì -1 -> îïðåäåëèòåëü ðàâåí 0 è âûõîäèì èç öèêëà
{
*strnum=-1;
}

}