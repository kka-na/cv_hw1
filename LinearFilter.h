#ifndef LinearFilter_h
#define LinearFilter_h

#include <iostream>
#include <iomanip>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

class LinearFilter{
public:
	
	double gaussianEQ(float x, float y, double sig){
		return exp( -(pow(x,2)+pow(y,2)) / (2*pow(sig,2)) ) / (2*CV_PI*pow(sig,2));
	}

	float myKernelConv(int dim, uchar* arr, float** kernel, int x, int y, int width, int height){
		float sum = 0;
		float sumKernel = 0;
		int cell = dim/2;
		for(int j = -cell; j<=cell; j++){
			for (int i=-cell; i<=cell; i++){
				if((y+j) >= 0 && (y+j) < height && (x+i) >=0 && (x+i) < width){
					sum += arr[(y+j)*width + (x+i)]*kernel[i+cell][j+cell];
					sumKernel += kernel[i+cell][j+cell];
				}
			}
		}
		if(sumKernel != 0){
			return sum/sumKernel;
		}else{
			return sum;
		}
	}

	float my3ChKernelConv(int dim, Mat arr, float** kernel, int x, int y, int width, int height, int idx){
		float sum = 0;
		float sumKernel = 0;
		int cell = dim/2;
		for(int j = -cell; j<=cell; j++){
			for (int i=-cell; i<=cell; i++){
				if((y+j) >= 0 && (y+j) < height && (x+i) >=0 && (x+i) < width){
					sum += arr.at<Vec3b>(y+j, x+i)[idx] *kernel[i+cell][j+cell];
					sumKernel += kernel[i+cell][j+cell];
				}
			}
		}
		if(sumKernel != 0){
			return sum/sumKernel;
		}else{
			return sum;
		}
	}

	Mat myGaussianFilter(Mat srcImg, int dim, bool show_kernel){
		int width = srcImg.cols;
		int height = srcImg.rows;

		float** kernel = new float*[dim];
		for(int i=0; i<dim; i++){
			kernel[i] = new float[dim];
		}

		for(int c=0; c<dim; c++){
			for(int r=0; r<dim; r++){
				kernel[c][r] = (float)gaussianEQ((float)(c-dim/2),(float)(r-dim/2), 1.0);
			}
		}

		if(show_kernel){
			for(int i=0; i<dim; i++){
				for(int j=0; j<dim; j++){
					cout<<kernel[i][j]<<" ";
				}
				cout<<endl;
			}
		}
		
		Mat dstImg(srcImg.size(), srcImg.type());
		uchar* srcData = srcImg.data;

		for(int y = 0; y<height; y++){
			for(int x=0; x<width; x++){
				if(srcImg.channels() == 1){
					dstImg.at<uchar>(y,x)= myKernelConv(dim,srcData, kernel, x,y,width, height);
				}else{
					dstImg.at<Vec3b>(y,x)[0] = my3ChKernelConv(dim, srcImg, kernel, x,y, width, height,0);
					dstImg.at<Vec3b>(y,x)[1] = my3ChKernelConv(dim, srcImg, kernel, x,y, width, height,1);
					dstImg.at<Vec3b>(y,x)[2] = my3ChKernelConv(dim, srcImg, kernel, x,y, width, height,2);
				}	
			}
		}

		for(int i=0; i<dim; i++){ delete[] kernel[i]; }
		delete[] kernel;

		return dstImg;

	}

	std::vector<Mat> myGaussianPyramid(Mat srcImg, int pyramid_num, int type){
		vector<Mat> Vec;
		Vec.push_back(srcImg);
		string window_name;
		for(int i=0; i<pyramid_num; i++){
			srcImg = mySampling(srcImg);
			srcImg = myGaussianFilter(srcImg,type,false);
			Vec.push_back(srcImg);
		}
		return Vec;
	}

	std::vector<Mat> myLaplacianPyramid(Mat srcImg, int pyramid_num, int type){
		vector<Mat> Vec;
		for(int i=0; i<pyramid_num; i++){
			Mat highImg = srcImg;
			srcImg = mySampling(srcImg);
			srcImg = myGaussianFilter(srcImg,type,false);
			Mat lowImg = srcImg;
			resize(lowImg, lowImg, highImg.size());
			// the images for the Laplacian pyramid are visualized by adding 0.5 ( 255/2 ~= 128 ), so light gray values are positive and dark gray values are negative.
			if(srcImg.channels() == 1){
				Vec.push_back(highImg - lowImg + 128);
			}else{
				for(int y=0; y<highImg.rows; y++){
					for(int x=0; x<highImg.cols; x++){
						for(int c=0; c<3; c++){
							highImg.at<Vec3b>(y,x)[c] = highImg.at<Vec3b>(y,x)[c] - lowImg.at<Vec3b>(y,x)[c] + 128;
						}
					}
				}
				Vec.push_back(highImg);
			}
		}
		return Vec;
	}

	Mat mySampling(Mat srcImg){
		int width = srcImg.cols/2;
		int height = srcImg.rows/2;
		Mat dstImg(height, width, srcImg.type());
		for(int y=0; y<height; y++){
			for(int x=0; x<width; x++){
				if(srcImg.channels() == 1)
					dstImg.at<uchar>(y,x) = srcImg.at<uchar>(2*y, 2*x);
				else{
					for(int i=0; i<3; i++){
						dstImg.at<Vec3b>(y,x)[i] = srcImg.at<Vec3b>(2*y, 2*x)[i];
					}
				}
			}
		}
		return dstImg;
	}
	

};
#endif /* LinearFilter_h */