class Query{
public:
__host__ __device__ Query(){
};
__host__ __device__ Query(int _min, int _max){
min = _min;
max = _max;
};
int min;
int max;
};
//new series 
