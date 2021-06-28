// Traveling Antsmen constants and struct declarations

#define MAX_CITIES 4462 // number of vertices
#define MAX_DIST 100000
#define MAX_TOUR (MAX_CITIES * MAX_DIST)
#define MAX_ANTS MAX_CITIES
#define NUM_EDGES ((MAX_CITIES * MAX_CITIES - MAX_CITIES) / 2)

struct cityType {
  int x, y;
};

// allows edge queries as a 2D array
class EdgeMatrix {
  float *dist;
public:
  EdgeMatrix() {
    dist = new float[MAX_CITIES * MAX_CITIES];
  }
  ~EdgeMatrix() {
    delete dist;
  }
  float* operator[](unsigned int i) {
    return &dist[MAX_CITIES * i];
  }

  float *get_array(){
    return dist;
  }
};

//Ant algorithm problem parameters
#define ALPHA 3.0
#define BETA 1.0 // this parameter raises the weight of distance over pheromone
#define RHO 0.1 // evaporation rate
#define QVAL 1
#define MAX_TOURS 20
#define MAX_TIME (MAX_TOURS * MAX_CITIES)
#define INIT_PHER (1.0 / MAX_CITIES)
