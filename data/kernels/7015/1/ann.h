#ifndef ANN_HEADER
#define ANN_HEADER

#include <helper_cuda.h>
#include <cmath>
#include <cstdlib>

#include <cmath>
#include <math.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <array>

#include <random>
using namespace std;
//
// Random
//
class Random {
  private:
    std::mt19937 *mGen;
    std::uniform_real_distribution<double> *mDist;

  public:
    Random();
    double next();
    int nextInt(int min, int max);
    bool nextBool();
};

class Topology {
	private:
		std::vector<int> *ml;
	public:
		Topology();
		~Topology();
		void addLayer(int size);

		int getLayerCount();
		int getLayerSize(int index);

		int obtainNeuronCount();
		int obtainWeightCount();

		int getInputNeuronCount();
		int getOutputNeuronCount();

    void printTopology(FILE *file);
    void readTopology(FILE *file);
};

struct Sample_Double{
	double * input;
	double * output;
};

struct Sample_Float{
	float * input;
	float * output;
};

class Data_Double{
  public:
  	int getNumberOfInputs() { return inputs; }
  	int getNumberOfOutputs() { return outputs; }

  	double * getInput(int index);

  	double * getOutput(int index);

  	int getNumberOfSamples() { return data.size(); }

  	void addSample(Sample_Double sample);

  	void setSizes(int input_size, int output_size);

  protected:
  	std::vector<Sample_Double> data;
    int inputs;
    int outputs;
};

class Data_Float{
  public:
  	int getNumberOfInputs() { return inputs; }
  	int getNumberOfOutputs() { return outputs; }

  	float * getInput(int index);

  	float * getOutput(int index);

  	int getNumberOfSamples() { return data.size(); }

  	void addSample(Sample_Float sample);

  	void setSizes(int input_size, int output_size);

  protected:
  	std::vector<Sample_Float> data;
  	int inputs;
  	int outputs;
};

template <typename T>
class AnnBase {
  public:
  	virtual void train(T *a, T *b, T alpha, T eta) = 0;
    virtual void finishTraining() = 0;
  	virtual void feedForward(T *a, T *b) = 0;
  	virtual void destroy() = 0;
  	virtual T obtainError(T *b) = 0;

  	virtual void print_out() = 0;

  private:
    virtual void prepare(Topology *top) = 0;
    virtual	void init(FILE *pFile)=0;
  	virtual void calc_feedForward() = 0;

};

class AnnSerialDBL : public AnnBase<double> {
  private:
  	Topology* cTopology;

  	int L;
  	int * l;
  	int * s;
  	double * a_arr;
  	double * z_arr;
  	int * W;
  	int * sw;
  	double * w_arr;
  	double * dw_arr;
  	double * t_arr;
  	double * gjl;
  public:
  	void train(double *a, double *b, double alpha, double eta);
    void finishTraining();
  	void feedForward(double *a, double *b);
  	void destroy();

  	double obtainError(double *b);
  	void print_out();

    void setWeights(double *t_w_arr){
      w_arr=t_w_arr;
    };

  	AnnSerialDBL(string filename) {
        FILE * p1File;
        p1File = fopen(filename.c_str(), "rb");
        Topology *top=new Topology();
        top->readTopology(p1File);
        prepare(top);
        init(p1File);
        fclose (p1File);
    };

    AnnSerialDBL(Topology *top) {
      prepare(top);
      init(NULL);
    };

  	double* getWeights();
    double* getDWeights();
  	double* getA();
    Topology* getTopology();

    void printf_Network(string filename);


  private:
    void prepare(Topology *top);
    void init(FILE *pFile);

  	void calc_feedForward();
  	double delta_w(double grad, double dw, double alpha, double eta);
  	double f(double x);
  	double f_deriv(double x);
  	double gL(double a, double z, double t);
  	double w_gradient(int layer_id, int w_i, int w_j);
  	void calc_gjl();

    void readf_Network(FILE *pFile);
};

class AnnSerialFLT : public AnnBase<float> {
  private:
  	Topology* cTopology;

  	int L;
  	int * l;
  	int * s;
  	float * a_arr;
  	float * z_arr;
  	int * W;
  	int * sw;
  	float * w_arr;
  	float * dw_arr;
  	float * t_arr;
  	float * gjl;

  public:
  	void train(float *a, float *b, float alpha, float eta);
    void finishTraining();
  	void feedForward(float *a, float *b);
  	void destroy();

  	float obtainError(float *b);
  	void print_out();

    void setWeights(float *t_w_arr){
      w_arr=t_w_arr;
    };

    AnnSerialFLT(Topology *top) {
      prepare(top);
      init(NULL);
    };

  	float* getWeights();

    void printf_Network(string filename);


  private:
    void prepare(Topology *top);
    void init(FILE *pFile);

  	void calc_feedForward();
  	float delta_w(float grad, float dw, float alpha, float eta);
  	float f(float x);
  	float f_deriv(float x);
  	float gL(float a, float z, float t);
  	float w_gradient(int layer_id, int w_i, int w_j);
  	void calc_gjl();
};

class AnnCUDA : public AnnBase<float> {
  private:
  	Topology* cTopology;

  	int L;
  	int * l;
  	int * s;
  	float * a_arr;
  	float * z_arr;
  	int * W;
  	int * sw;
  	float * w_arr;
  	float * dw_arr;
  	float * t_arr;
  	float * gjl;

    int *dv_l; int bc_l;
    int *dv_s; int bc_s;

    float *dv_a_arr; int bc_a_arr;
    float *dv_z_arr; int bc_z_arr;

    int *dv_W; int bc_W;
    int *dv_sw; int bc_sw;

    float *dv_w_arr; int bc_w_arr;
    float *dv_dw_arr; int bc_dw_arr;

    float *dv_t_arr; int bc_t_arr;
    float *dv_gjl; int bc_gjl;


  public:
  	void train(float *a, float *b, float alpha, float eta);
    void finishTraining();
  	void feedForward(float *a, float *b);
  	void destroy();

  	float obtainError(float *b);
  	void print_out();

    void setWeights(float *t_w_arr);

    AnnCUDA(Topology *top) {
      prepare(top);
      init(NULL);
    };

  	float* getWeights();

    void printf_Network(string filename);


  private:
    void prepare(Topology *top);
    void init(FILE *pFile);

  	void calc_feedForward();
  	float delta_w(float grad, float dw, float alpha, float eta);
  	float f(float x);
  	float f_deriv(float x);
  	float gL(float a, float z, float t);
  	float w_gradient(int layer_id, int w_i, int w_j);
  	void calc_gjl();
};


//*****************Cuda 2.0***************************

class AnnCUDA2 : public AnnBase<float> {
  private:
  	Topology* cTopology;
    int h;
    int h2;
  	int L;
  	int * l;
    int * l_ext;
  	int * s_ext;
  	float * a_ext_arr;
  	float * z_ext_arr;
  	int * sw_ext;

  	float * w_ext_arr;
  	float * dw_ext_arr;
  	float * t_arr;
  	float * gjl_ext;

    int *dv_l; int bc_l;
    int *dv_s_ext; int bc_s_ext;

    float *dv_a_ext_arr; int bc_a_ext_arr;
    float *dv_z_ext_arr; int bc_z_ext_arr;


    int *dv_sw_ext; int bc_sw_ext;

    float *dv_w_ext_arr; int bc_w_ext_arr;
    float *dv_dw_ext_arr; int bc_dw_ext_arr;

    float *dv_t_arr; int bc_t_arr;
    float *dv_gjl_ext; int bc_gjl_ext;


  public:
  	void train(float *a, float *b, float alpha, float eta);
    void finishTraining();
  	void feedForward(float *a, float *b);
  	void destroy();

  	float obtainError(float *b);
  	void print_out();

    void setWeights(float *t_w_arr);

    AnnCUDA2(Topology *top) {
      prepare(top);
      init(NULL);
    };

  	float* getWeights();
    float* getA();

    void printf_Network(string filename);


  private:
    void prepare(Topology *top);
    void init(FILE *pFile);

  	void calc_feedForward();

  	void calc_gjl();

  public:
    static int obtainNeuronCountExt(Topology *top);
    static int obtainWeightCountExt(Topology *top);
};

/* Class definitions here. */
void run_cuda_sample();

#endif /* ANN_HEADER */
