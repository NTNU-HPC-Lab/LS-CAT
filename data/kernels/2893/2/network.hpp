#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector> 
#ifndef mrobi100_network
#define mrobi100_network

#define MAX_LAYERS 4
#define DEFAULT_ALPHA .1
#define DEFAULT_EPOCHS 100

struct NetworkArch{
    int inputLayer; 
    int layer1; 
    int layer2;
    int outputLayer; 
} typedef NetworkArch;

struct Network{
    float* w1; 
    float* w2; 
    float* w3; 
} typedef Network;

struct NetworkOuput{
    float* layer1; 
    float* layer2; 
    float* output; 
} typedef NetworkOutput;


using namespace std;
class InputValues {
    public: 
        bool useValidationSet = false;
        bool training = false;
        bool gt = false;
        bool usePredefWeights = false;
        bool performEvalutation = false;
        std::string archFile;
        std::string weightsFile; 
        std::string trainingFile; 
        std::string gtFile;
        std::string validationFile;
        std::string evaluationFile; 
        std::string outputFile; 
        int epochs = -1;
        float alpha = -1;
        
    
        void readInputValues(int argc, char** argv){
            //check for no inputs, and output help message if so.
            checkHelp(argc);
            for(int i = 0; i < argc; i++){

                std::string input(argv[i]);

                if (!input.compare("--archFile"))
                {
                    this->archFile = std::string(argv[++i]);
                }
                else if (!input.compare("--weights"))
                {
                    this->weightsFile = std::string(argv[++i]);
                }
                else if (!input.compare("--training"))
                {
                    this->trainingFile = std::string(argv[++i]);
                }
                else if (!input.compare("--groundTruth"))
                {
                    this->gtFile = std::string(argv[++i]);
                }
                else if (!input.compare("--validation"))
                {
                    this->validationFile = std::string(argv[++i]);
                }
                else if (!input.compare("--evaluation"))
                {
                    this->evaluationFile = std::string(argv[++i]);
                }
                else if (!input.compare("--output"))
                {
                    this->outputFile = std::string(argv[++i]);
                }
                else if (!input.compare("--epochs"))
                {
                    this->epochs = stoi(std::string(argv[++i]));
                }
                else if (!input.compare("--alpha"))
                {
                    this->alpha = stof(std::string(argv[++i]));
                }
            }
        }

        void validateArgs(){
            if(archFile.empty()){
                cout << "No arch file specified, Exiting" << endl;
                exit(EXIT_FAILURE);
            }
            usePredefWeights = !weightsFile.empty();
            training = !trainingFile.empty();
            gt = !gtFile.empty();
            performEvalutation = !evaluationFile.empty();
            useValidationSet = !validationFile.empty();

            if(training && !gt || gt && !training){
                cout << "Must provide both training data and associated ground truth" << endl;
                exit(EXIT_FAILURE);
            }

            if(!training && !performEvalutation){
                cout << "User must specify if training or performing evaluation or both" << endl;
                exit(EXIT_FAILURE);
            }

            if(alpha == -1){
                alpha = DEFAULT_ALPHA;
            }

            if(epochs <= 0){
                epochs = DEFAULT_EPOCHS;
            }
        }

        void checkHelp(int argc){
            if(argc <= 1){
                cout << "Usage is: ./network.exe --archFile <> --weights <optional> --training <trainingDataFile> --groundTruth <gtFile> --evaluation <dataFileForEval> --output <networkWeightSaveFile> --alpha <.1> --epochs <200>" << endl;
                exit(EXIT_SUCCESS);
            }
        }

};


NetworkArch* readNetworkArch(InputValues* iv){
    NetworkArch* networkArch = new NetworkArch;
    string archFile = iv->archFile;
    ifstream f (archFile);
    if (!f.good()){
        cout<< "Bad Arch File" << endl;
        exit(EXIT_FAILURE);
    }    

    string layerSize; 
    getline(f, layerSize, ',');
    networkArch->inputLayer = stoi(layerSize);
    
    getline(f, layerSize, ',');
    networkArch->layer1 = stoi(layerSize);
    
    getline(f, layerSize, ',');
    networkArch->layer2 = stoi(layerSize);
    
    getline(f, layerSize); // get last layer number
    networkArch->outputLayer = stoi(layerSize);

    f.close();

    return networkArch;
};

void readWeightsFile(string weightsFile, Network** network_ref, NetworkArch* networkArch){
    // Network* network_h = new Network; 
    Network* network_h = *network_ref;
    ifstream f (weightsFile);
    if (!f.good()){
        cout<< "Bad Weights File" << endl;
        exit(EXIT_FAILURE);
    }  

    // read the three layers of weights
    
    float* w1 = (float*) malloc(networkArch->inputLayer*networkArch->layer1 *sizeof(float));
    float* w2 = (float*) malloc(networkArch->layer1 * networkArch->layer2 *sizeof(float));
    float* w3 = (float*) malloc(networkArch->layer2*networkArch->outputLayer *sizeof(float));
    
    string line1;
    string line2;
    string line3;
    getline(f, line1);
    getline(f, line2);
    getline(f, line3);
    
    stringstream ss(line1);
    for(int i = 0; i < networkArch->inputLayer*networkArch->layer1; i++){
        string floatValue;
        getline(ss, floatValue, ',');
        w1[i] = std::stof(floatValue);
    }

    stringstream ss1(line2);
    for(int i = 0; i < networkArch->layer1 * networkArch->layer2 ; i++){
        string floatValue;
        getline(ss1, floatValue, ',');
        w2[i] = std::stof(floatValue);
    }

    stringstream ss2(line3);
    for(int i = 0; i < networkArch->layer2*networkArch->outputLayer; i++){
        string floatValue;
        getline(ss2, floatValue, ',');
        w3[i] = std::stof(floatValue);
    }

    network_h->w1 = w1;
    network_h->w2 = w2;
    network_h->w3 = w3;

    f.close();
}


void readData(string filePath, vector<float*>* data, int elements_per_line){
    ifstream f (filePath);
    if (!f.good()){
        cout<< "Bad File" << endl;
        exit(EXIT_FAILURE);
    }  

    // read data in
    while(f.good()){

        float* values = (float*) malloc(elements_per_line*sizeof(float));
        
        string line;
        getline(f, line);
        
        stringstream ss(line);
        for(int i = 0; i < elements_per_line; i++){
            string floatValue;
            getline(ss, floatValue, ',');
            // cout << floatValue << endl;
            if(!floatValue.compare("") || !floatValue.compare("\n")){
                free(values);
                break;
            }
            values[i] = std::stof(floatValue);
        }
        (*data).push_back(values);
    }
    f.close();
}

void writeWeights(string filePath, NetworkArch* networkArch, Network* network){
    ofstream f (filePath);
    if (!f.good()){
        cout<< "Bad file to write weights too." << endl;
        exit(EXIT_FAILURE);
    }  

    // write csv    
    int numWeights = (networkArch->inputLayer) * (networkArch->layer1);
    float* w = network->w1;
    for(int i = 0; i < numWeights; i++){
        if(i != numWeights-1){
            f << w[i] << ",";
        }else {
            f << w[i];
        }
    }
    f<<endl;

    numWeights = (networkArch->layer1) * (networkArch->layer2);
    w = network->w2;
    for(int i = 0; i < numWeights; i++){
        if(i != numWeights-1){
            f << w[i] << ",";
        }else {
            f << w[i];
        }
    }
    f<<endl;

    numWeights = (networkArch->layer2) * (networkArch->outputLayer);
    w = network->w3;
    for(int i = 0; i < numWeights; i++){
        if(i != numWeights-1){
            f << w[i] << ",";
        }else {
            f << w[i];
        }
    }
    // do not put a newline at the end of the file
    f.close();
}

void writeResultData(string filePath, vector<float*>* results, int elements_per_line){
    ofstream f (filePath);
    vector<float*> res = *results;
    for(int j = 0; j < res.size(); j++){
        float * res_line = res[j];
        for(int i = 0; i < elements_per_line; i++){
            if(i != elements_per_line-1){
                f << res_line[i] << ",";
            }else {
                f << res_line[i];
            }
        }
        f << endl;
    }
    f.close();
}

void printAvgTraingTimes(int totalItters, float total_itter_time, float total_fp_time, float total_bp_time){
    //print times
    printf("Average Times of compute: \n");
    printf("Iteration: %.4fms Forward Pass: %.4fms BackProp: %.4fms \n", total_itter_time/totalItters, total_fp_time/totalItters, total_bp_time/totalItters);
    cout << endl;
}

#endif
