#include "includes.h"

int answersNumber;
int categoriesNumber;
int atribsNumber;

/**
* Funkcja wykonywana na karcie graficznej - kazdy watek sprawdza czy jego atrybut z atribsValues to ten sam co w query. Jesli tak, przepisuje do
* tablicy wynikowej prawdopodobiestwa dla kazdej jego odpowiedzi
* @param query - zapytanie uzytkownika w postacie zlepionych stringow
* @param atribsValues - tablica wszystkich atrybutow
* @param possibilities - tablica wszystkich prawdopodobienstw
* @param queryPrefix - tablica sum prefiksowych dlugosci slow w query
* @param atribsPrefix - j.k. dla atribsValues
* @param answersNumber - liczba mozliwych odpowiedzi
* @param categoriesNumber - liczba kategorii
* @param atribsNumber - liczba wszystkich atrybutow
* @param resultPossibilities - tablica prawdopodobienstw atrybutow z zapytania dla wszystkich mozliwych odpowiedzi
*/

__global__ void searchWithCuda(double *resultPossibilities, char *query, char *atribsValues, double *possibilities, int *queryPrefix, int *atribsPrefix, int *answersNumber, int *categoriesNumber, int *atribsNumber)
{
int category_id = blockIdx.x;	// categories
int atrib_id = blockIdx.y;	// atribs

// znajdz poczatek lancucha znakow atrybutu w zapytaniu i w atribsValue
char *queryAtrib = query + queryPrefix[category_id];
int queryAtribLength = queryPrefix[category_id + 1] - queryPrefix[category_id];

char *currAtrib = atribsValues + atribsPrefix[atrib_id];
int currAtribLength = atribsPrefix[atrib_id + 1] - atribsPrefix[atrib_id];

if (queryAtribLength == currAtribLength)
{
bool equal = true;
for (int i = 0; i < queryAtribLength; ++i)
{
if (queryAtrib[i] != currAtrib[i])
{
equal = false;
break;
}
}
if (equal)	// przypisz odpowiednie prawdopodobienstwa
{
for (int i = 0; i < *answersNumber; ++i)
{
resultPossibilities[*categoriesNumber*i + category_id] = possibilities[*atribsNumber*i + atrib_id];	// na razie tylko dla jednej odpowiedzi
}
}
}
}