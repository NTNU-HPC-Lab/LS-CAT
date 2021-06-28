#include <iostream>
#include <vector>
#include <tuple>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp> 

using namespace std;

vector<int*> getImageAsIntArrays(cv::Mat& img) {

	vector<int*> ret;
	for(int i=0;i<3;i++)
		ret.push_back(new int[img.rows*img.cols]);

	for(int i=0;i<img.cols;i++) {
		for(int j=0;j<img.rows;j++) {
			/*B*/ret[0][j*img.cols+i] = img.at<cv::Vec3b>(j,i)[0] ;
			/*G*/ret[1][j*img.cols+i] = img.at<cv::Vec3b>(j,i)[1] ;
			/*R*/ret[2][j*img.cols+i] = img.at<cv::Vec3b>(j,i)[2] ;
		}
	}

	return ret;
}

tuple< vector<int*> , int , int > getIntArrays(char* s) {
	cv::Mat image;
	image = cv::imread(s, cv::IMREAD_COLOR);
	if( image.empty() )
		exit(EXIT_FAILURE);

	return make_tuple(getImageAsIntArrays(image),image.rows,image.cols);
}

bool saveImage(vector<int*> channels,int rows,int cols) {
	cv::Mat img(rows,cols,CV_8UC3,cv::Scalar(0,0,0));

	for(int i=0;i<img.cols;i++) {
		for(int j=0;j<img.rows;j++) {
			/*B*/img.at<cv::Vec3b>(j,i)[0] = channels[0][j*img.cols+i] ;
			/*G*/img.at<cv::Vec3b>(j,i)[1] = channels[1][j*img.cols+i] ;
			/*R*/img.at<cv::Vec3b>(j,i)[2] = channels[2][j*img.cols+i] ;
		}
	}

	

	cv::imwrite("./result.jpg",img);

	return true;
}