#include <vector>

#include "LinearFilter.h"

Mat makeDisplayforPyramid(vector<Mat> Vec, Mat orig, int size_div);
Mat makeDisplayforGauLap(vector<Mat> Vec, Mat orig, int size_div);

int main(){
    cout<<"\nComputer Vision HW1\n22212231 김가나 과제\n";
	int question_num = 0;
	cout<<"\n\n፨ 문제 번호 입력. ፨"<<endl;
	cout<<"[ Question #1]\nGaussian/Laplacian pyramid\n";
	cout<<"[ Question #2]\nHybrid image\n";
	
    LinearFilter lf; 

    Mat img_man = imread("../image/man.jpg", IMREAD_COLOR); //IMREAD_COLOR or IMREAD_GRAYSCALE;
    Mat img_dog = imread("../image/dog.jpg", IMREAD_COLOR); //IMREAD_COLOR or IMREAD_GRAYSCALE;
    Mat img_cat = imread("../image/cat.jpg", IMREAD_COLOR); //IMREAD_COLOR or IMREAD_GRAYSCALE;


    while(true){
		cout<<"\n፨ Quit to Enter 0 ፨"<<endl;
		cout<<"▶▷▶ ";
		cin>>question_num;
        if(question_num == 1){
            Mat img_man_clone = img_man.clone();
            Mat img_gaussian = lf.myGaussianFilter(img_man_clone, 5, true);
            Mat res_gaussian_filter;
            hconcat(img_man_clone, img_gaussian, res_gaussian_filter);
            imshow("Man image 5x5 Gaussian Filter", res_gaussian_filter);
            imwrite("../result/man_5x5_gaussian.jpg", res_gaussian_filter);
            
            std::vector<Mat> VecGauss = lf.myGaussianPyramid(img_man_clone, 8, 9);
            std::vector<Mat> VecLap = lf.myLaplacianPyramid(img_man_clone, 8, 9);
            Mat res;
            vconcat(makeDisplayforPyramid(VecGauss, img_man_clone, 4), makeDisplayforGauLap(VecLap, img_man_clone, 4), res);
            imshow("Man Image 9x9 Gaussian Pyramid & Laplacian Pyramid Result", res);
            imwrite("../result/man_9x9_pyramids.jpg", res);

        }
        else if(question_num == 2){
            Mat img_dog_clone = img_dog.clone();
            Mat img_cat_clone = img_cat.clone();
            resize(img_cat_clone, img_cat_clone, img_dog_clone.size());

            std::vector<Mat> VecGauss_dog = lf.myGaussianPyramid(img_dog_clone, 4, 3);
            std::vector<Mat> VecGauss_cat = lf.myGaussianPyramid(img_cat_clone, 4, 3);
            std::vector<Mat> VecLap_dog = lf.myLaplacianPyramid(img_dog_clone, 4, 3);
            std::vector<Mat> VecLap_cat = lf.myLaplacianPyramid(img_cat_clone, 4, 3);
            Mat dog_res, cat_res;
            vconcat(makeDisplayforPyramid(VecGauss_dog, img_dog_clone, 2), makeDisplayforGauLap(VecLap_dog, VecLap_dog[0], 2), dog_res);
            vconcat(makeDisplayforPyramid(VecGauss_cat, img_cat_clone, 2), makeDisplayforGauLap(VecLap_cat, VecLap_cat[0], 2), cat_res);
            imshow("Dog Image 3x3 Guassian Pyramid & Laplacian Pyramid Results",dog_res);
            imwrite("../result/dog_3x3_pyramids.jpg", dog_res);

            imshow("Cat Image 3x3 Guassian Pyramid & Laplacian Pyramid Results",  cat_res);
            imwrite("../result/cat_3x3_pyramids.jpg", cat_res);

            Mat res = Mat::zeros(img_dog_clone.size(), img_dog_clone.type());
            resize(VecLap_dog[3], VecLap_dog[3], img_dog_clone.size());
            resize(VecLap_cat[0], VecLap_cat[0], img_cat_clone.size());

            if(img_dog_clone.channels() == 1){
                res = VecLap_dog[3]+VecLap_cat[0]-128;
            }else{
                for(int y=0; y<img_dog_clone.rows; y++){
                    for(int x=0; x<img_dog_clone.cols; x++){
                        for(int c=0; c<3; c++){
                            res.at<Vec3b>(y,x)[c] = VecLap_dog[3].at<Vec3b>(y,x)[c] + VecLap_cat[0].at<Vec3b>(y,x)[c] - 128;
                        }
                    }
                }
            }
            imshow("Hybrid Image of Cat & Dog", res);
            imwrite("../result/cat_dog_hybrid.jpg", res);

        }
        
        else if(question_num == 0){
			cout<<"End HW1 ... "<<endl<<endl;
			break;
		}else{
			cout<<"Enter Nubmer again"<<endl;
		}
		waitKey(0);
		destroyAllWindows();
	}
	return 0;
	
}

Mat makeDisplayforPyramid(vector<Mat> Vec, Mat orig, int size_div){
    Size quater = orig.size()/size_div;
    Mat res;
	for(int i=0; i<Vec.size(); i++){
        resize(Vec[i], Vec[i], quater);
    }
    hconcat(Vec[0], Vec[1], res);
    for(int i=2; i<Vec.size(); i++){
        hconcat(res, Vec[i], res);
    }
	return res;
}


Mat makeDisplayforGauLap(vector<Mat> Vec, Mat orig, int size_div){
    Size quater = orig.size()/size_div;
    Mat black = Mat::zeros(orig.rows/size_div, (orig.cols/size_div)/2, orig.type());
    Mat res;
	for(int i=0; i<Vec.size(); i++){
        resize(Vec[i], Vec[i], quater);
    }
    hconcat(black,Vec[0],res);
    for(int i=1; i<Vec.size(); i++){
        hconcat(res, Vec[i], res);   
    }
    hconcat(res, black, res);
    if((orig.cols/size_div) % 2 != 0){
        hconcat(res, Mat::zeros(orig.rows/size_div, 1, orig.type()), res);
    }
    return res;
}