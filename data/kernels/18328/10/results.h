#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

using namespace std;

class Performance_results{
public:
	char filename[200];
	double GPU_FFT_time;
	double GPU_MSD_time;
	double GPU_HRMS_time;
	double GPU_total_time;
	double GPU_stdev;
	int nElements;
	int nHarmonics;
	int nSeries;
	int nRuns;
	int device;
	
	Performance_results() {
		GPU_FFT_time = 0;
		GPU_MSD_time = 0;
		GPU_HRMS_time = 0;
		GPU_total_time = 0;
		GPU_stdev = 0;
		nElements = 0;
		nHarmonics = 0;
		nSeries = 0;
		nRuns = 0;
		device = 0;
	}
	
	void Save(){
		ofstream FILEOUT;
		FILEOUT.open (filename, std::ofstream::out | std::ofstream::app);
		FILEOUT << std::fixed << std::setprecision(8) << nElements << " " << nHarmonics << " " << nSeries << " " << nRuns << " " << GPU_FFT_time  << " " << GPU_MSD_time  << " " << GPU_HRMS_time  << " " << GPU_total_time << " " << GPU_stdev << endl;
		FILEOUT.close();
	}
	
	void Print(){
		cout << std::fixed << std::setprecision(8) << nElements << " " << nHarmonics << " " << nSeries << " " << nRuns << " " << GPU_FFT_time  << " " << GPU_MSD_time  << " " << GPU_HRMS_time  << " " << GPU_total_time << " " << GPU_stdev << endl;
	}
	
	void Assign(int t_nElements, int t_nHarmonics, int t_nSeries, int t_nRuns, const char *t_filename){
		nElements  = t_nElements;
		nHarmonics = t_nHarmonics;
		nSeries    = t_nSeries;
		nRuns      = t_nRuns;

		sprintf(filename,"%s", t_filename);
	}
	
};
