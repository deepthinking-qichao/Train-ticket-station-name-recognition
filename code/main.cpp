/************************************************************
Filename: Ticket_Detection
Author: Wu Qichao
Version; 5_0_6
Date: 2017-10-29
Description:1）20171029 revise
************************************************************/
/***************************/
/*    header file include  */
/*    macro definition     */
/*        namesapce        */
/***************************/
#include <opencv2/core/core.hpp>  
#include<opencv2/highgui/highgui.hpp>  
#include<opencv2/imgproc/imgproc.hpp>  
#include <opencv2/ml/ml.hpp>  
#include <iostream>  
#include <vector>
#include <string>
#include<fstream>  
#include <stdio.h>
#include <stdlib.h>

#define GAUSSIANBLUR_SIZE 5
#define SOBEL_X_WEIGHT 1
#define SOBEL_Y_WEIGHT 0.5
#define MORPH_SIZE_WIDTH 21			//25			
#define MORPH_SIZE_HEIGHT 2			//2			
#define selective_picture 1
#define SOBEL_SCALE 2
#define SOBEL_DELTA 0
#define SOBEL_DDEPTH CV_16S
#define minarea 500				
#define height_threshold 25		
#define angle_threshold 20
#define duty_cycle_threshold 0.3	
#define aspect_ratio_threshold 0.8
#define duty_cycle_char 0.2
#define aspect_ratio_threshold_min_char 0.1
#define aspect_ratio_threshold_char_max 4
#define Height_min_char 20 
#define Height_max_char 80
#define area_threshold_char 100
//#define CHAR_SIZE 20
#define MORPH_SIZE_WIDTH_CHAR 1
#define MORPH_SIZE_HEIGHT_CHAR 23    
#define char_binary_threshold 5				
#define HORIZONTAL    1
#define VERTICAL    0
#define Angle_Scale 3
#define hough_line_threshold 170	
#define hough_line_dec 5			//4
#define grid_width 5
#define grid_height 7
#define resize_width 30
#define resize_height 42
#define s_height 2

using namespace std;
using namespace cv;

/************************/
/*	 struct and class 	*/
/************************/
struct output
{
	int num;
	string info_t;
	string info_s1;
	string info_s2;
	string info_d;
	string info_p;
};

/************************/
/*	 global variable 	*/
/************************/
//ticket adjustment
int line_threshold = hough_line_threshold;
int canny_high_threshold = 60;
int canny_low_threshold = 30;

//information selection 
int info_t_num = 2;
int info_s1_num = 4;
int info_s2_num = 5;
int info_d_num = 6;
int info_p_num = 7;

//char identify
const char strchar_num_t[] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '(', ')', '*' };
const char strchar_capital_t[] = { 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', \
'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', \
'U', 'V', 'W', 'X', 'Y', 'Z' };
const char strchar_char_s[] = { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', \
'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', \
'u', 'v', 'w', 'x', 'y', 'z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', \
'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', \
'u', 'v', 'w', 'x', 'y', 'z' };
const char strchar_num_d[] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*' };
const char strchar_num_p[] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '*', '*', '*', '*', '*' };
const char strchar_lowercase_p[] = { 's', 'z', 'x', 'z' };

//debug flag
//if you want to show information during debugging, please set debug1=1 and debug2=2
//if you want to show time information during processing, please set debug3=1 and debug4=4
//if you want to see the base-result, you must set debug5=1 and debug6=1
//if you want to see adjustment mid-process output, you must set debug_adjust=1
#define debug1 1					//text of debug information								0  
#define debug2 1					//picture of debug information							0
#define debug3 0					//time of program process								0
#define debug4 0					//time of char recognition								0
#define debug5 1					//debug of key in/out information						1
#define debug6 1					//debug of picture adujustment error contral			1	
#define debug_adjust 1				//debug of picture adjustment mid-process output		0

//time
double time0;	//debug time for each period
double time1;   //debug time for char recognition
double time2;   //debug time for each char recognition
double time3;   //debug time for train num recognition time
double time4;	//debug time for each ticket 

//others
int count_pic = 0;

//bayes model
bool load_bayes_flag = false;
CvNormalBayesClassifier normalBayes_capital_t;
CvNormalBayesClassifier normalBayes_num_t;
CvNormalBayesClassifier normalBayes_char_s;
CvNormalBayesClassifier normalBayes_num_d;
CvNormalBayesClassifier normalBayes_lowercase_p;
CvNormalBayesClassifier normalBayes_num_p;

/************************************************************************/
/*	 Generate horizontal projection and vertical projection histogram 	*/
/************************************************************************/
Mat ProjectedHistogram1(const Mat &img, Rect &box)
{
	int sz = img.rows;
	Mat mhist = Mat::zeros(1, sz, CV_32F);
	int min_y = 0;
	int max_y = img.rows - 1;
	bool min_flag = false;
	bool max_flag = false;

	for (int j = 0; j <= sz - 1; j++)
	{
		Mat data = img.row(j);
		mhist.at<float>(j) = countNonZero(data);
		if ((mhist.at<float>(j) > 0) && (!min_flag))
		{
			min_y = j;
			min_flag = true;
		}

	}

	for (int j = sz - 1; j >= 0; j--)
	{
		Mat data = img.row(j);
		mhist.at<float>(j) = countNonZero(data);
		if ((mhist.at<float>(j) > 0) && (!max_flag))
		{
			max_y = j;
			max_flag = true;
		}
	}

	box.y = min_y;
	box.height = max_y - min_y + 1;
	return mhist;
}

Mat ProjectedHistogram2(const Mat &img, Rect &box)
{
	int sz = img.cols;
	Mat mhist = Mat::zeros(1, sz, CV_32F);
	int min_x = 0;
	int max_x = img.cols - 1;
	bool min_flag = false;
	bool max_flag = false;

	for (int j = 0; j <= sz - 1; j++)
	{
		Mat data = img.col(j);
		mhist.at<float>(j) = countNonZero(data);
		if ((mhist.at<float>(j) > 0) && (!min_flag))
		{
			min_x = j;
			min_flag = true;
		}

	}

	for (int j = sz - 1; j >= 0; j--)
	{
		Mat data = img.col(j);
		mhist.at<float>(j) = countNonZero(data);
		if ((mhist.at<float>(j) > 0) && (!max_flag))
		{
			max_x = j;
			max_flag = true;
		}
	}

	box.x = min_x;
	box.width = max_x - min_x + 1;
	return mhist;
}

Mat ProjectedHistogram_r(const Mat &img)
{
	int sz = img.rows;
	Mat mhist = Mat::zeros(1, sz, CV_32F);

	for (int j = 0; j <= sz - 1; j++)
	{
		Mat data = img.row(j);
		mhist.at<float>(j) = countNonZero(data) / float(img.cols);
	}
	return mhist;
}

Mat ProjectedHistogram_c(const Mat &img)
{
	int sz = img.cols;
	Mat mhist = Mat::zeros(1, sz, CV_32F);

	for (int j = 0; j <= sz - 1; j++)
	{
		Mat data = img.col(j);
		mhist.at<float>(j) = countNonZero(data) / float(img.rows);
	}
	return mhist;
}

/**********************************/
/*	 Generating a trimming image  */
/**********************************/
vector<Mat> cut_pic(const Mat &in, const int &j, int &p_sum)
{
	vector<Mat> cut_pic_sep;
#if (debug1)
		printf("\n图像的原始宽为%d，原始高为%d", in.cols, in.rows);
#endif
	Rect box;
	Mat vhist = ProjectedHistogram1(in, box);			//0
	Mat hhist = ProjectedHistogram2(in, box);
#if (debug1)
		printf("左上角为（%d，%d）,宽为%d，高为%d\n", box.x, box.y, box.width, box.height);
#endif
	Mat roi;
	roi = in(Rect(box.x, box.y, box.width, box.height));
	int area = roi.rows * roi.cols;
	if (j == 3)
	{
		if (area < 250)
		{
			return cut_pic_sep;
		}
	}
	if ((j == 1) || (j == 2) || (j == 3) || (j == 4))
	//if ((j == 1) || (j == 2) || (j == 3))
	{
		if ((roi.cols > 0) && (roi.cols <= 25))
		{
#if (debug1)
			printf("没有粘连\n");
#endif
			cut_pic_sep.push_back(roi.clone());
			p_sum += 1;
		}
		if ((roi.cols > 25) && (roi.cols <= 50))
		{
			int r = roi.cols / 2;
			if (debug1)
				printf("有两个粘连，左上角的x坐标分别为0,%d\n", r);
			cut_pic_sep.push_back(roi(Rect(0, 0, r + 1, roi.rows)));
			cut_pic_sep.push_back(roi(Rect(r, 0, roi.cols - r, roi.rows)));
			//cut_pic_sep[0] = roi(Rect(0, 0, r + 1, roi.rows));
			//cut_pic_sep[1] = roi(Rect(r, 0, roi.cols - r, roi.rows));
			p_sum += 2;
		}
		if ((roi.cols > 50) && (roi.cols <= 75))
		{
			int r = roi.cols / 3;
			if (debug1)
				printf("有三个粘连，左上角的x坐标分别为0,%d,%d\n", r, 2 * r);
			cut_pic_sep.push_back(roi(Rect(0, 0, r + 1, roi.rows)));
			cut_pic_sep.push_back(roi(Rect(r, 0, r + 1, roi.rows)));
			cut_pic_sep.push_back(roi(Rect(2 * r, 0, roi.cols - 2 * r, roi.rows)));
			//cut_pic_sep[0] = roi(Rect(0, 0, r + 1, roi.rows));
			//cut_pic_sep[1] = roi(Rect(r, 0, r + 1, roi.rows));
			//cut_pic_sep[2] = roi(Rect(2 * r, 0, roi.cols - r, roi.rows));
			p_sum += 3;
		}
		if ((roi.cols > 75) && (roi.cols <= 110))
		{
			int r = roi.cols / 4;
			if (debug1)
				printf("有四个粘连，分割点的x坐标分别为0,%d,%d,%d,%d\n", r, 2 * r, 3 * r, roi.cols - 1);
			cut_pic_sep.push_back(roi(Rect(0, 0, r + 1, roi.rows)));
			cut_pic_sep.push_back(roi(Rect(r, 0, r + 1, roi.rows)));
			cut_pic_sep.push_back(roi(Rect(2 * r, 0, r + 1, roi.rows)));
			cut_pic_sep.push_back(roi(Rect(3 * r, 0, roi.cols - 3 * r, roi.rows)));
			p_sum += 4;
		}
		if (roi.cols > 110)
		{
			cut_pic_sep.push_back(roi.clone());
		}
	}
	else
	{
		cut_pic_sep.push_back(roi.clone());
	}

	//return out;
	return cut_pic_sep;
}

/************************/
/*	 Get mesh features  */
/************************/
bool get_grid_feature(const Mat &img_threshold_resize, Mat &trainData)
{
	//Mat img_threshold_resize;
	//resize(img_threshold, img_threshold_resize, Size(resize_width, resize_height), 0, 0, INTER_CUBIC);
	int r = img_threshold_resize.rows / grid_height;
	int c = img_threshold_resize.cols / grid_width;
	float sum = 0;
	int k = 0;
	for (int i = 0; i <= grid_height * r - 1; i = i + r)
	{
		for (int j = 0; j <= grid_width * c - 1; j = j + c)
		{
			for (int m = i; m <= i + r - 1; m++)
			{
				for (int n = j; n <= j + c - 1; n++)
				{
					if ((m < img_threshold_resize.rows) && (n < img_threshold_resize.cols))
					{
						if (img_threshold_resize.at<uchar>(m, n) == 255)
							sum++;
					}
				}
			}
			trainData.at<float>(0, k) = sum / float(r*c);
			sum = 0;
			k++;
		}
	}
	Mat mhist_r = ProjectedHistogram_r(img_threshold_resize);
	for (int i = 0; i < img_threshold_resize.rows; i++)
	{
		trainData.at<float>(0, k) = mhist_r.at<float>(i);
		k++;
	}
	Mat mhist_c = ProjectedHistogram_c(img_threshold_resize);
	for (int i = 0; i < img_threshold_resize.cols; i++)
	{
		trainData.at<float>(0, k) = mhist_c.at<float>(i);
		k++;
	}
	return true;
}

/***************************/
/*	character recognition  */
/***************************/
string charsIdentify(const Mat &input, vector<Mat> &qie_bian_pic, const int &j, const int &k, Mat &szx, const int &num, int &p_sum, const CvNormalBayesClassifier &testNbc_num, const CvNormalBayesClassifier &testNbc_num_szx)
{
	time2 = (double)getTickCount();
	qie_bian_pic = cut_pic(input, j, p_sum);
	for (int l = 0; l < qie_bian_pic.size(); l++)
	{
		//resize(qie_bian_pic[l], qie_bian_pic[l], Size(resize_width, resize_height));
		resize(qie_bian_pic[l], qie_bian_pic[l], Size(resize_width, resize_height), 0, 0, INTER_CUBIC);
		//Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
		//erode(qie_bian_pic[l], qie_bian_pic[l], element);
	}
#if (debug4)
	{
		cout << "\t\t qie bian time is ";
		cout << ((double)getTickCount() - time2) / (double)getTickFrequency() << endl;
		time2 = (double)getTickCount();
	}
#endif
	//string result = "";
	string bayes_result = "";

	/*识别列车号信息*/
	if (j == 0)
	{
		if (k == 0)
		{
			for (int l = 0; l < qie_bian_pic.size(); l++)
			{
				time3 = (double)getTickCount();
				Mat testSample(1, grid_width*grid_height + resize_width + resize_height, CV_32FC1);

				imwrite("test_t.bmp", qie_bian_pic[l]);
#if (debug4)
				{
					cout << "\t\t\t write jpg time is ";
					cout << ((double)getTickCount() - time3) / (double)getTickFrequency() << endl;
					time3 = (double)getTickCount();
				}
#endif
				Mat tmp1 = imread("test_t.bmp", 0);
#if (debug4)
				{
					cout << "\t\t\t read jpg time is ";
					cout << ((double)getTickCount() - time3) / (double)getTickFrequency() << endl;
					time3 = (double)getTickCount();
				}
#endif
				Mat img_threshold;
				threshold(tmp1, img_threshold, 0, 255, CV_THRESH_BINARY);	//注意读取的图片是否为单通道图
#if (debug4)
				{
					cout << "\t\t\t binary time is ";
					cout << ((double)getTickCount() - time3) / (double)getTickFrequency() << endl;
					time3 = (double)getTickCount();
				}
#endif
				get_grid_feature(img_threshold, testSample);
#if (debug4)
				{
					cout << "\t\t\t get feature time is ";
					cout << ((double)getTickCount() - time3) / (double)getTickFrequency() << endl;
					time3 = (double)getTickCount();
				}
#endif
				float flag = testNbc_num.predict(testSample);
				int index_num = flag;
				bayes_result = bayes_result + strchar_capital_t[index_num];
#if (debug4)
				{
					cout << "\t\t\t predict time is ";
					cout << ((double)getTickCount() - time3) / (double)getTickFrequency() << endl;
					time3 = (double)getTickCount();
				}
#endif
			}
		}
		else
		{
			for (int l = 0; l < qie_bian_pic.size(); l++)
			{
				time3 = (double)getTickCount();
				Mat testSample(1, grid_width*grid_height + resize_width + resize_height, CV_32FC1);

				imwrite("test_t.bmp", qie_bian_pic[l]);
#if (debug4)
				{
					cout << "\t\t\t write jpg time is ";
					cout << ((double)getTickCount() - time3) / (double)getTickFrequency() << endl;
					time3 = (double)getTickCount();
				}
#endif
				Mat tmp1 = imread("test_t.bmp", 0);
#if (debug4)
				{
					cout << "\t\t\t read jpg time is ";
					cout << ((double)getTickCount() - time3) / (double)getTickFrequency() << endl;
					time3 = (double)getTickCount();
				}
#endif
				Mat img_threshold;
				threshold(tmp1, img_threshold, 0, 255, CV_THRESH_BINARY);	//注意读取的图片是否为单通道图
#if (debug4)
				{
					cout << "\t\t\t binary time is ";
					cout << ((double)getTickCount() - time3) / (double)getTickFrequency() << endl;
					time3 = (double)getTickCount();
				}
#endif
				get_grid_feature(img_threshold, testSample);
#if (debug4)
				{
					cout << "\t\t\t get feature time is ";
					cout << ((double)getTickCount() - time3) / (double)getTickFrequency() << endl;
					time3 = (double)getTickCount();
				}
#endif
				//get_grid_feature(qie_bian_pic[l], testSample);

				float flag = testNbc_num.predict(testSample);
				int index_num = flag;
				bayes_result = bayes_result + strchar_num_t[index_num];
#if (debug4)
				{
					cout << "\t\t\t predict time is ";
					cout << ((double)getTickCount() - time3) / (double)getTickFrequency() << endl;
					time3 = (double)getTickCount();
				}
#endif
			}
		}
	}
#if (debug4)
	{
		cout << "\t\t train num recognition time is ";
		cout << ((double)getTickCount() - time2) / (double)getTickFrequency() << endl;
		time2 = (double)getTickCount();
	}
#endif
	/*识别站名信息*/
	if ((j == 1) || (j == 2))
	{
		for (int l = 0; l < qie_bian_pic.size(); l++)
		{
			time3 = (double)getTickCount();
			Mat testSample(1, grid_width*grid_height + resize_width + resize_height, CV_32FC1);

			imwrite("test_s.bmp", qie_bian_pic[l]);
#if (debug4)
			{
				cout << "\t\t\t write jpg time is ";
				cout << ((double)getTickCount() - time3) / (double)getTickFrequency() << endl;
				time3 = (double)getTickCount();
			}
#endif
			Mat tmp1 = imread("test_s.bmp", 0);
#if (debug4)
			{
				cout << "\t\t\t read jpg time is ";
				cout << ((double)getTickCount() - time3) / (double)getTickFrequency() << endl;
				time3 = (double)getTickCount();
			}
#endif
			Mat img_threshold;
			threshold(tmp1, img_threshold, 0, 255, CV_THRESH_BINARY);	//注意读取的图片是否为单通道图
#if (debug4)
			{
				cout << "\t\t\t binary time is ";
				cout << ((double)getTickCount() - time3) / (double)getTickFrequency() << endl;
				time3 = (double)getTickCount();
			}
#endif
			get_grid_feature(img_threshold, testSample);
#if (debug4)
			{
				cout << "\t\t\t get feature time is ";
				cout << ((double)getTickCount() - time3) / (double)getTickFrequency() << endl;
				time3 = (double)getTickCount();
			}
#endif
			float flag = testNbc_num.predict(testSample);
			int index_num = flag;
			bayes_result = bayes_result + strchar_char_s[index_num];
#if (debug4)
			{
				cout << "\t\t\t predict time is ";
				cout << ((double)getTickCount() - time3) / (double)getTickFrequency() << endl;
				time3 = (double)getTickCount();
			}
#endif
		}
	}
#if (debug4)
	{
		cout << "\t\t station recognition time is ";
		cout << ((double)getTickCount() - time2) / (double)getTickFrequency() << endl;
		time2 = (double)getTickCount();
	}
#endif
	/*识别日期信息*/
	if (j == 3)
	{
		for (int l = 0; l < qie_bian_pic.size(); l++)
		{
			time3 = (double)getTickCount();
			Mat testSample(1, grid_width*grid_height + resize_width + resize_height, CV_32FC1);

			imwrite("test_d.bmp", qie_bian_pic[l]);
#if (debug4)
			{
				cout << "\t\t\t write jpg time is ";
				cout << ((double)getTickCount() - time3) / (double)getTickFrequency() << endl;
				time3 = (double)getTickCount();
			}
#endif
			Mat tmp1 = imread("test_d.bmp", 0);
#if (debug4)
			{
				cout << "\t\t\t read jpg time is ";
				cout << ((double)getTickCount() - time3) / (double)getTickFrequency() << endl;
				time3 = (double)getTickCount();
			}
#endif
			Mat img_threshold;
			threshold(tmp1, img_threshold, 0, 255, CV_THRESH_BINARY);	//注意读取的图片是否为单通道图
#if (debug4)
			{
				cout << "\t\t\t binary time is ";
				cout << ((double)getTickCount() - time3) / (double)getTickFrequency() << endl;
				time3 = (double)getTickCount();
			}
#endif
			get_grid_feature(img_threshold, testSample);
#if (debug4)
			{
				cout << "\t\t\t get feature time is ";
				cout << ((double)getTickCount() - time3) / (double)getTickFrequency() << endl;
				time3 = (double)getTickCount();
			}
#endif

			//get_grid_feature(qie_bian_pic[l], testSample);		
			//cout << testSample << endl;

			float flag = testNbc_num.predict(testSample);
			int index_num = flag;
			bayes_result = bayes_result + strchar_num_d[index_num];
#if (debug4)
			{
				cout << "\t\t\t predict time is ";
				cout << ((double)getTickCount() - time3) / (double)getTickFrequency() << endl;
				time3 = (double)getTickCount();
			}
#endif
		}
	}
#if (debug4)
	{
		cout << "\t\t date recognition time is ";
		cout << ((double)getTickCount() - time2) / (double)getTickFrequency() << endl;
		time2 = (double)getTickCount();
	}
#endif
	/*识别座位信息*/
	if (j == 4)
	{
		if ((p_sum > 7) && (k == num - 2))
		{
			time3 = (double)getTickCount();
			Mat testSample(1, grid_width*grid_height + resize_width + resize_height, CV_32FC1);

			vector<Mat> cut_tmp = cut_pic(szx, 0, p_sum);
			resize(cut_tmp[0], cut_tmp[0], Size(resize_width, resize_height), 0, 0, INTER_CUBIC);
			imwrite("test_p0.bmp", cut_tmp[0]);
#if (debug4)
			{
				cout << "\t\t\t write jpg time is ";
				cout << ((double)getTickCount() - time3) / (double)getTickFrequency() << endl;
				time3 = (double)getTickCount();
			}
#endif
			Mat tmp1 = imread("test_p0.bmp", 0);
#if (debug4)
			{
				cout << "\t\t\t read jpg time is ";
				cout << ((double)getTickCount() - time3) / (double)getTickFrequency() << endl;
				time3 = (double)getTickCount();
			}
#endif
			Mat img_threshold;
			threshold(tmp1, img_threshold, 0, 255, CV_THRESH_BINARY);	//注意读取的图片是否为单通道图
#if (debug4)
			{
				cout << "\t\t\t binary time is ";
				cout << ((double)getTickCount() - time3) / (double)getTickFrequency() << endl;
				time3 = (double)getTickCount();
			}
#endif
			get_grid_feature(img_threshold, testSample);
#if (debug4)
			{
				cout << "\t\t\t get feature time is ";
				cout << ((double)getTickCount() - time3) / (double)getTickFrequency() << endl;
				time3 = (double)getTickCount();
			}
#endif

			//get_grid_feature(qie_bian_pic[l], testSample);

			float flag = testNbc_num_szx.predict(testSample);
			int index_num = flag;
			bayes_result = bayes_result + strchar_lowercase_p[index_num];
#if (debug4)
			{
				cout << "\t\t\t predict time is ";
				cout << ((double)getTickCount() - time3) / (double)getTickFrequency() << endl;
				time3 = (double)getTickCount();
			}
#endif
		}

		if (p_sum <= 7)
		{
			for (int l = 0; l < qie_bian_pic.size(); l++)
			{
				time3 = (double)getTickCount();
				Mat testSample(1, grid_width*grid_height + resize_width + resize_height, CV_32FC1);

				imwrite("test_p.bmp", qie_bian_pic[l]);
#if (debug4)
				{
					cout << "\t\t\t write jpg time is ";
					cout << ((double)getTickCount() - time3) / (double)getTickFrequency() << endl;
					time3 = (double)getTickCount();
				}
#endif
				Mat tmp1 = imread("test_p.bmp", 0);
#if (debug4)
				{
					cout << "\t\t\t read jpg time is ";
					cout << ((double)getTickCount() - time3) / (double)getTickFrequency() << endl;
					time3 = (double)getTickCount();
				}
#endif
				Mat img_threshold;
				threshold(tmp1, img_threshold, 0, 255, CV_THRESH_BINARY);	//注意读取的图片是否为单通道图
#if (debug4)
				{
					cout << "\t\t\t binary time is ";
					cout << ((double)getTickCount() - time3) / (double)getTickFrequency() << endl;
					time3 = (double)getTickCount();
				}
#endif
				get_grid_feature(img_threshold, testSample);
#if (debug4)
				{
					cout << "\t\t\t get feature time is ";
					cout << ((double)getTickCount() - time3) / (double)getTickFrequency() << endl;
					time3 = (double)getTickCount();
				}
#endif

				//get_grid_feature(qie_bian_pic[l], testSample);

				float flag = testNbc_num.predict(testSample);
				int index_num = flag;
				bayes_result = bayes_result + strchar_num_p[index_num];
#if (debug4)
				{
					cout << "\t\t\t predict time is ";
					cout << ((double)getTickCount() - time3) / (double)getTickFrequency() << endl;
					time3 = (double)getTickCount();
				}
#endif
			}
		}
	}
#if (debug4)
	{
		cout << "\t\t position recognition time is ";
		cout << ((double)getTickCount() - time2) / (double)getTickFrequency() << endl;
		time2 = (double)getTickCount();
	}
#endif
	return bayes_result;
}

/****************************/
/*	 Text block selection 	*/
/****************************/
bool verifySizes(const RotatedRect &mr, const float &duty_cycle)
{
	float r = (float)mr.size.width / (float)mr.size.height;
	float area = mr.size.height * mr.size.width;

	if ((area > minarea) && (r > aspect_ratio_threshold) && (mr.size.height >height_threshold) && (duty_cycle >duty_cycle_threshold) && (mr.angle - angle_threshold < 0) && (mr.angle + angle_threshold > 0))
	{
		return true;
	}
	else
	{
		return false;
	}
}

/***************************************/
/*	 get text block from source image  */
/***************************************/
Mat showResultMat(const Mat &src, const Size &rect_size, const Point2f &center)
{
	Mat img_crop;
	getRectSubPix(src, rect_size, center, img_crop);
	return img_crop;
}

/************************/
/*	 sort text block	*/
/************************/
int SortRect_pic(const vector<RotatedRect>& vecRect, vector<RotatedRect>& out)
{
	vector<int> orderIndex;
	vector<int> block_positions;
	int tmp;

	for (int i = 0; i < vecRect.size(); i++)
	{
		orderIndex.push_back(i);
		tmp = vecRect[i].center.x + vecRect[i].center.y * 20;		//y*10
		block_positions.push_back(tmp);
	}

	float min = block_positions[0];
	int minIdx = 0;
	for (int i = 0; i< block_positions.size(); i++)
	{
		min = block_positions[i];
		minIdx = i;
		for (int j = i; j<block_positions.size(); j++)
		{
			if (block_positions[j]<min){
				min = block_positions[j];
				minIdx = j;
			}
		}
		int aux_i = orderIndex[i];
		int aux_min = orderIndex[minIdx];
		orderIndex[i] = aux_min;
		orderIndex[minIdx] = aux_i;

		float aux_xi = block_positions[i];
		float aux_xmin = block_positions[minIdx];
		block_positions[i] = aux_xmin;
		block_positions[minIdx] = aux_xi;
	}

	for (int i = 0; i<orderIndex.size(); i++)
	{
		out.push_back(vecRect[orderIndex[i]]);
	}

	return 0;
}

/************************/
/*	text block segment 	*/
/************************/
int text_block_segment(const Mat &src, vector<Mat>& resultVec, Mat &img_adjust_sub_resize1, vector<Mat>& resultVec_5, const int &picnum)
{
	Mat src_blur, src_gray;
	Mat grad;

	int scale = SOBEL_SCALE;
	int delta = SOBEL_DELTA;
	int ddepth = SOBEL_DDEPTH;

	if (!src.data)
	{
		cout << "none of picture inputed to process in text_block_segment funciton" << endl;
		return -1;
	}

	Mat srcclone = src.clone();
	//高斯均衡。Size中的数字影响车牌定位的效果。
	GaussianBlur(src, src_blur, Size(GAUSSIANBLUR_SIZE, GAUSSIANBLUR_SIZE), 0, 0, BORDER_DEFAULT);

	/// Convert it to gray
	cvtColor(src_blur, src_gray, CV_RGB2GRAY);

	/// Generate grad_x and grad_y
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	/// Gradient X
	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	/// Gradient Y
	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	/// Total Gradient (approximate)
	addWeighted(abs_grad_x, SOBEL_X_WEIGHT, abs_grad_y, SOBEL_Y_WEIGHT, 0, grad);
#if (debug2)
		imshow("sobel", grad);					//对图像进行Sobel运算，得到的是图像的一阶水平方向导数
#endif 

	Mat img_threshold;
	threshold(grad, img_threshold, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
#if (debug2)
		imshow("二值化图像", img_threshold);
#endif

	/*erase interfere of 2 shorter size of the red ticket */
	Mat element_c1 = getStructuringElement(MORPH_RECT, Size(2, 1));
	Mat element_o1 = getStructuringElement(MORPH_RECT, Size(5, 1));
	morphologyEx(img_threshold, img_threshold, CV_MOP_CLOSE, element_c1);
#if (debug2)
		imshow("闭操作后的图像pre", img_threshold);
#endif
	morphologyEx(img_threshold, img_threshold, CV_MOP_OPEN, element_o1);
#if (debug2)
		imshow("开操作后的图像", img_threshold);
#endif

	Mat element = getStructuringElement(MORPH_RECT, Size(MORPH_SIZE_WIDTH, MORPH_SIZE_HEIGHT));
	morphologyEx(img_threshold, img_threshold, CV_MOP_CLOSE, element);
#if (debug2)
		imshow("闭操作后的图像", img_threshold);
#endif

	//Find contours of possibles plates
	Mat img_threshold_clone = img_threshold.clone();
	vector< vector< Point> > contours;
	findContours(img_threshold_clone,
		contours, // a vector of contours
		CV_RETR_EXTERNAL, // extract external contours
		CV_CHAIN_APPROX_NONE); // all pixels of each contours

	Mat contours_detect = Mat::zeros(src.rows, src.cols, CV_8UC3);
#if (debug2)
	{
		drawContours(contours_detect, contours, -1, Scalar(255, 255, 255), 1, 8);
		imshow("轮廓图", contours_detect);
	}
#endif
	Mat contours_detect_clone = contours_detect.clone();

	//Start to iterate to each contour founded
	vector<vector<Point> >::iterator itc = contours.begin();

	vector<RotatedRect> rects;
	//Remove patch that are no inside limits of aspect ratio and area.
	int t = 0;
	while (itc != contours.end())
	{
		//Create bounding rect of object
		RotatedRect mr = minAreaRect(Mat(*itc));
		float contour_area = fabs(contourArea(*itc));

#if (debug2)
		{
			Point2f vertices[4];
			mr.points(vertices);
			for (int i = 0; i < 4; i++)
			{
				line(contours_detect, vertices[i], vertices[(i + 1) % 4], Scalar(0, 0, 255));
			}
		}
#endif

		/*print text tag*/
#if (debug2)
		{
			string words = to_string(t);
			putText(contours_detect, words, mr.center, CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255));
			imshow("未筛选的矩形图-轮廓图", contours_detect);
		}
#endif

		float r = (float)mr.size.width / (float)mr.size.height;
		float angle = mr.angle;
		Size rect_size = mr.size;
		float area = (float)mr.size.width * (float)mr.size.height;
		float duty_cycle = contour_area / area;			//定义占空比，为轮廓面积除以最小外界矩形面积

#if (debug1)
			printf("第%d个未筛选的可旋转矩形的角度为%f,长宽比为%f，面积为%f,宽为%f，高为%f, 占空比为%f\n", t, angle, r, area, mr.size.width, mr.size.height, duty_cycle);
#endif

		/*make sure which size is width*/
		float angle_correcting_0 = fabs(mr.angle);
		float angle_correcting_90 = fabs(fabs(mr.angle) - 90);							//校正矩形旋转角度，统一矩形的长宽对应的实际长度
		bool angle_flag = angle_correcting_0 > angle_correcting_90;
		if (angle_flag)
		{
			if (mr.angle < 0)
				mr.angle = mr.angle + 90;
			else
				mr.angle = mr.angle - 90;
			swap(mr.size.width, mr.size.height);
			r = (float)mr.size.width / (float)mr.size.height;
		}

#if (debug1)
			printf("第%d个调整后的未筛选的可旋转矩形的角度为%f,长宽比为%f，面积为%f,宽为%f，高为%f, 占空比为%f\n", t, mr.angle, r, area, mr.size.width, mr.size.height, duty_cycle);
#endif

		//根据预设条件（长宽比，高，占空比等）进行文字行区域筛选
		if (!verifySizes(mr, duty_cycle))
		{
			itc = contours.erase(itc);
		}
		else
		{
			++itc;
			rects.push_back(mr);
		}
		t++;
	}

#if (debug1)
	{
		printf("\n\n有%d个未筛选的矩形\t", t);
		printf("有%d个筛选后的矩形\n\n", rects.size());
	}
#endif

	if (rects.size() == 0)
	{
		cout << "can not find any text block rectangle" << endl;
		return -1;
	}

	/*sort text block*/
	vector<RotatedRect> sortedRect;
	SortRect_pic(rects, sortedRect);
#if (debug1)
		printf("\n文字行区域排序完成\n");
#endif

	vector<Vec4f> center_five;
	vector<Vec4f> center_five_tmp;
	Vec4f center_tmp;
	//vector<Vec4b> info_t_flag;
	vector<Mat> resultVec_5_tmp;
	vector<Mat> resultVec_5_tmp_s;
	//const int tmp1 = sortedRect.size();

	//bool flag_t5 = false;

	for (int i = 0; i< sortedRect.size(); i++)
	{
		RotatedRect minRect = sortedRect[i];

		float r = (float)minRect.size.width / (float)minRect.size.height;
		float angle = minRect.angle;
		Size rect_size = minRect.size;
		float area = (float)minRect.size.width * (float)minRect.size.height;

		Size rect_size_s = minRect.size;
		rect_size_s.height += 2 * s_height;
		Point2f center_s = minRect.center;
		center_s.y -= s_height;

		//if angle > 30 false
		if (1)
		{
#if (debug1)
				printf("第%d个筛选后的可旋转矩形的角度为%f,长宽比为%f，面积为%f,宽为%f，高为%f\n", i, angle, r, area, minRect.size.width, minRect.size.height);
#endif
#if (debug2)
			{
				Point2f vertices[4];
				minRect.points(vertices);
				for (int i = 0; i < 4; i++)
				{
					line(img_threshold, vertices[i], vertices[(i + 1) % 4], Scalar(0, 0, 255));
					line(srcclone, vertices[i], vertices[(i + 1) % 4], Scalar(0, 0, 255));
					line(contours_detect_clone, vertices[i], vertices[(i + 1) % 4], Scalar(0, 0, 255));
				}
				string words = to_string(i);
				putText(contours_detect_clone, words, minRect.center, CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255));
				putText(srcclone, words, minRect.center, CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255));
				imshow("筛选后的矩形图-二值图", img_threshold);
				imshow("筛选后的矩形图-原图", srcclone);
				img_adjust_sub_resize1 = srcclone.clone();
				imshow("筛选后的矩形图-轮廓图", contours_detect_clone);
			}
#endif

			//Create and rotate image
			Mat rotmat = getRotationMatrix2D(minRect.center, angle, 1);
			Mat img_rotated;
			warpAffine(src, img_rotated, rotmat, src.size(), CV_INTER_CUBIC);

			Mat resultMat;
			resultMat = showResultMat(img_rotated, rect_size, minRect.center);

			Mat resultMat_s;
			resultMat_s = showResultMat(img_rotated, rect_size_s, center_s);

			resultVec.push_back(resultMat);

			resultVec_5_tmp.push_back(resultMat);
			resultVec_5_tmp_s.push_back(resultMat_s);

			center_tmp[0] = minRect.size.width;				//width
			center_tmp[1] = minRect.size.height;			//height
			center_tmp[2] = minRect.center.x;				//center.x
			center_tmp[3] = minRect.center.y;				//center.y
#if (debug1)
				printf("\n第%d个宽，高，中心为%f,%f,%f,%f\n", i, center_tmp[0], center_tmp[1], center_tmp[2], center_tmp[3]);
#endif
			center_five_tmp.push_back(center_tmp);
		}
	}

	/*match train number*/
	bool info_t_flag[50][4];
	int info_t_flag_all[50];
	for (int i = 0; i < sortedRect.size(); i++)
	{
		for (int j = 0; j < 4; j++)
		{
			info_t_flag[i][j] = false;
		}
	}
	bool flag_t1 = false;
	bool flag_t2 = false;
	bool flag_t3 = false;
	bool flag_t4 = false;
	int train_num = 0;

	for (int i = 0; i < sortedRect.size(); i++)
	{
#if (debug1)
			printf("第%d个车次宽，高，中心横坐标，中心纵坐标是%f,%f,%f,%f\n", i, center_five_tmp[i][0], center_five_tmp[i][1], center_five_tmp[i][2], center_five_tmp[i][3]);
#endif

		if ((center_five_tmp[i][0] > 100) && (center_five_tmp[i][0] < 180))
			info_t_flag[i][0] = true;
		if ((center_five_tmp[i][1] > 39) && (center_five_tmp[i][1] < 45))		
			info_t_flag[i][1] = true;
		if ((center_five_tmp[i][2] > 390) && (center_five_tmp[i][2] < 470))
			info_t_flag[i][2] = true;
		if ((center_five_tmp[i][3] > 100) && (center_five_tmp[i][3] < 160))
			info_t_flag[i][3] = true;

		info_t_flag_all[i] = int(info_t_flag[i][0]) + int(info_t_flag[i][1]) + int(info_t_flag[i][2]) + int(info_t_flag[i][3]);
							
#if (debug1)
			printf("第%d个列车号标志是%d,%d,%d,%d\n", i, info_t_flag[i][0], info_t_flag[i][1], info_t_flag[i][2], info_t_flag[i][3]);
#endif
	} 

	for (int i = 0; i < sortedRect.size(); i++)
	{
		if (info_t_flag_all[i] == 4)
		{
			center_five.push_back(center_five_tmp[i]);
			resultVec_5.push_back(resultVec_5_tmp[i]);
			flag_t1 = true;
			train_num = i;
			break;
		}
	}

	if (flag_t1 == false)
	{
		for (int i = 0; i < sortedRect.size(); i++)
		{
			if (info_t_flag_all[i] == 3)
			{
				center_five.push_back(center_five_tmp[i]);
				resultVec_5.push_back(resultVec_5_tmp[i]);
				flag_t2 = true;
				train_num = i;
				break;
			}
		}
	}

	if ((flag_t1 == false) && (flag_t2 == false))
	{
		for (int i = 0; i < sortedRect.size(); i++)
		{
			if (info_t_flag_all[i] == 2)
			{
				center_five.push_back(center_five_tmp[i]);
				resultVec_5.push_back(resultVec_5_tmp[i]);
				flag_t3 = true;
				train_num = i;
				break;
			}
		}
	}

	if ((flag_t1 == false) && (flag_t2 == false) && (flag_t3 == false))
	{
		for (int i = 0; i < sortedRect.size(); i++)
		{
			if (info_t_flag_all[i] == 1)
			{
				center_five.push_back(center_five_tmp[i]);
				resultVec_5.push_back(resultVec_5_tmp[i]);
				flag_t4 = true;
				train_num = i;
				break;
			}
		}
	}

	if ((flag_t1 == false) && (flag_t2 == false) && (flag_t3 == false) && (flag_t4 == false))
	{
		cout << "train number can not found!!" << endl;
		return -1;
	}

	int sel_t = 0;
	if (flag_t1)
		sel_t = 1;
	else
	{
		if (flag_t2)
			sel_t = 0.5;
	}

	/*match station one*/
	bool info_s1_flag[50][4];
	int info_s1_flag_all[50];
	for (int i = 0; i < sortedRect.size(); i++)
	{
		for (int j = 0; j < 4; j++)
		{
			info_s1_flag[i][j] = false;
		}
	}
	bool flag_s11 = false;
	bool flag_s12 = false;
	bool flag_s13 = false;
	bool flag_s14 = false;
	int station1_num = 0;

	for (int i = 0; i < sortedRect.size(); i++)
	{
#if (debug1)
			printf("第%d个左站名宽，高，中心横坐标差，中心纵坐标差是%f,%f,%f,%f\n", i, center_five_tmp[i][0], center_five_tmp[i][1], center_five_tmp[i][2] - center_five_tmp[train_num][2], center_five_tmp[i][3] - center_five_tmp[train_num][3]);
#endif
		if ((center_five_tmp[i][0] > 50) && (center_five_tmp[i][0] < 280))		
			info_s1_flag[i][0] = true;	
		if ((center_five_tmp[i][1] > 25) && (center_five_tmp[i][1] < 40))		
			info_s1_flag[i][1] = true;
		if ((center_five_tmp[i][2] - center_five_tmp[train_num][2] > -290) && (center_five_tmp[i][2] - center_five_tmp[train_num][2]< -210))			
			info_s1_flag[i][2] = true;
		if ((center_five_tmp[i][3] - center_five_tmp[train_num][3]> 25) && (center_five_tmp[i][3] - center_five_tmp[train_num][3]< 70))				
			info_s1_flag[i][3] = true;
				
		info_s1_flag_all[i] = int(info_s1_flag[i][1]) + int(info_s1_flag[i][2]) + int(info_s1_flag[i][3]);
		
#if (debug1)
			printf("第%d个左站名标志是%d,%d,%d,%d\n", i, info_s1_flag[i][0], info_s1_flag[i][1], info_s1_flag[i][2], info_s1_flag[i][3]);
#endif
	}

	for (int i = 0; i < sortedRect.size(); i++)
	{
		if (i == train_num)
			continue;
		if ((info_s1_flag[i][0]) && (info_s1_flag_all[i] == 3))
		{
			center_five.push_back(center_five_tmp[i]);
			resultVec_5.push_back(resultVec_5_tmp_s[i]);

			flag_s11 = true;
			station1_num = i;
			break;
		}
	}

	if (flag_s11 == false)
	{
		for (int i = 0; i < sortedRect.size(); i++)
		{
			if (i == train_num)
				continue;
			if (info_s1_flag_all[i] == 3)
			{
				center_five.push_back(center_five_tmp[i]);
				resultVec_5.push_back(resultVec_5_tmp_s[i]);
				flag_s12 = true;
				station1_num = i;
				break;
			}
		}
	}

	if ((flag_s11 == false) && (flag_s12 == false))
	{
		for (int i = 0; i < sortedRect.size(); i++)
		{
			if (i == train_num)
				continue;
			if ((info_s1_flag[i][0]) && (info_s1_flag_all[i] == 2))
			{
				center_five.push_back(center_five_tmp[i]);
				resultVec_5.push_back(resultVec_5_tmp_s[i]);
				flag_s13 = true;
				station1_num = i;
				break;
			}
		}
	}

	if ((flag_s11 == false) && (flag_s12 == false) && (flag_s13 == false))
	{
		for (int i = 0; i < sortedRect.size(); i++)
		{
			if (i == train_num)
				continue;
			if ((info_s1_flag_all[i] == 2))
			{
				center_five.push_back(center_five_tmp[i]);
				resultVec_5.push_back(resultVec_5_tmp_s[i]);
				flag_s14 = true;
				station1_num = i;
				break;
			}
		}
	}

	if ((flag_s11 == false) && (flag_s12 == false) && (flag_s13 == false) && (flag_s14 == false))
	{
		cout << "station 1 can not found !!" << endl;
		return -1;
	}

	int sel_s1 = 0;
	if (flag_s11 || flag_s12)
		sel_s1 = 1;
	else
	{
		if (flag_s13)
			sel_s1 = 0.5;
	}

	/*matching station two*/
	bool info_s2_flag[50][4];
	int info_s2_flag_all[50];
	for (int i = 0; i < sortedRect.size(); i++)
	{
		for (int j = 0; j < 4; j++)
		{
			info_s2_flag[i][j] = false;
		}
	}
	bool flag_s21 = false;
	bool flag_s22 = false;
	bool flag_s23 = false;
	bool flag_s24 = false;
	int station2_num;

	for (int i = 0; i < sortedRect.size(); i++)
	{
#if (debug1)
			printf("第%d个右站名宽，高，中心横坐标差，中心纵坐标差是%f,%f,%f,%f\n", i, center_five_tmp[i][0], center_five_tmp[i][1], center_five_tmp[i][2] - center_five_tmp[train_num][2], center_five_tmp[i][3] - center_five_tmp[train_num][3]);
#endif
		if ((center_five_tmp[i][0] > 50) && (center_five_tmp[i][0] < 280))
			info_s2_flag[i][0] = true;
		if ((center_five_tmp[i][1] > 25) && (center_five_tmp[i][1] < 40))
			info_s2_flag[i][1] = true;
		if ((center_five_tmp[i][2] - center_five_tmp[train_num][2] > 210) && (center_five_tmp[i][2] - center_five_tmp[train_num][2]< 290))
			info_s2_flag[i][2] = true;
		if ((center_five_tmp[i][3] - center_five_tmp[train_num][3]> 25) && (center_five_tmp[i][3] - center_five_tmp[train_num][3]< 70))
			info_s2_flag[i][3] = true;

		info_s2_flag_all[i] = int(info_s2_flag[i][1]) + int(info_s2_flag[i][2]) + int(info_s2_flag[i][3]);
 
#if (debug1)
			printf("第%d个右站名标志是%d,%d,%d,%d\n", i, info_s2_flag[i][0], info_s2_flag[i][1], info_s2_flag[i][2], info_s2_flag[i][3]);
#endif
	}

	for (int i = 0; i < sortedRect.size(); i++)
	{
		if ((i == train_num) || (i == station1_num))
			continue;
		if ((info_s2_flag[i][0]) && (info_s2_flag_all[i] == 3))
		{
			center_five.push_back(center_five_tmp[i]);
			resultVec_5.push_back(resultVec_5_tmp_s[i]);
			flag_s21 = true;
			station2_num = i;
			break;
		}
	}

	if (flag_s21 == false)
	{
		for (int i = 0; i < sortedRect.size(); i++)
		{
			if ((i == train_num) || (i == station1_num))
				continue;
			if (info_s2_flag_all[i] == 3)
			{
				center_five.push_back(center_five_tmp[i]);
				resultVec_5.push_back(resultVec_5_tmp_s[i]);
				flag_s22 = true;
				station2_num = i;
				break;
			}
		}
	}

	if ((flag_s21 == false) && (flag_s22 == false))
	{
		for (int i = 0; i < sortedRect.size(); i++)
		{
			if ((i == train_num) || (i == station1_num))
				continue;
			if ((info_s2_flag[i][0]) && (info_s2_flag_all[i] == 2))
			{
				center_five.push_back(center_five_tmp[i]);
				resultVec_5.push_back(resultVec_5_tmp_s[i]);
				flag_s23 = true;
				station2_num = i;
				break;
			}
		}
	}

	if ((flag_s21 == false) && (flag_s22 == false) && (flag_s23 == false))
	{
		for (int i = 0; i < sortedRect.size(); i++)
		{
			if ((i == train_num) || (i == station1_num))
				continue;
			if (info_s2_flag_all[i] == 2)
			{
				center_five.push_back(center_five_tmp[i]);
				resultVec_5.push_back(resultVec_5_tmp_s[i]);
				flag_s24 = true;
				station2_num = i;
				break;
			}
		}
	}

	if ((flag_s21 == false) && (flag_s22 == false) && (flag_s23 == false) && (flag_s24 == false))
	{
		cout << "station 2 can not found !!" << endl;
		return -1;
	}

	int sel_s2 = 0;
	if (flag_s21 || flag_s22)
		sel_s2 = 1;
	else
	{
		if (flag_s23)
			sel_s2 = 0.5;
	}

	/*matich date*/
	bool info_d_flag[50][4];
	int info_d_flag_all[50];
	for (int i = 0; i < sortedRect.size(); i++)
	{
		for (int j = 0; j < 4; j++)
		{
			info_d_flag[i][j] = false;
		}
	}
	bool flag_d1 = false;
	bool flag_d2 = false;
	bool flag_d3 = false;
	bool flag_d4 = false;
	int d_num;

	for (int i = 0; i < sortedRect.size(); i++)
	{
#if (debug1)
			printf("第%d个日期宽，高，中心横坐标差，中心纵坐标差是%f,%f,%f,%f\n", i, center_five_tmp[i][0], center_five_tmp[i][1], center_five_tmp[i][2] - center_five_tmp[train_num][2], center_five_tmp[i][3] - center_five_tmp[train_num][3]);
#endif
		if ((center_five_tmp[i][0] > 380) && (center_five_tmp[i][0] < 470))
			info_d_flag[i][0] = true;
		if ((center_five_tmp[i][1] > 30) && (center_five_tmp[i][1] < 50))
			info_d_flag[i][1] = true;
		if ((center_five_tmp[i][2] - center_five_tmp[train_num][2] > -210) && (center_five_tmp[i][2] - center_five_tmp[train_num][2]< -130))
			info_d_flag[i][2] = true;
		if ((center_five_tmp[i][3] - center_five_tmp[train_num][3]> 60) && (center_five_tmp[i][3] - center_five_tmp[train_num][3]< 120))
			info_d_flag[i][3] = true;

		info_d_flag_all[i] = int(info_d_flag[i][0]) + int(info_d_flag[i][1]) + int(info_d_flag[i][2]) + int(info_d_flag[i][3]);

#if (debug1)
			printf("第%d个日期标志是%d,%d,%d,%d\n", i, info_d_flag[i][0], info_d_flag[i][1], info_d_flag[i][2], info_d_flag[i][3]);
#endif
	}

	for (int i = 0; i < sortedRect.size(); i++)
	{
		if ((i == train_num) || (i == station1_num) || (i == station2_num))
			continue;
		if (info_d_flag_all[i] == 4)
		{
			center_five.push_back(center_five_tmp[i]);
			resultVec_5.push_back(resultVec_5_tmp[i]);
			flag_d1 = true;
			d_num = i;
			break;
		}
	}

	if (flag_d1 == false)
	{
		for (int i = 0; i < sortedRect.size(); i++)
		{
			if ((i == train_num) || (i == station1_num) || (i == station2_num))
				continue;
			if (info_d_flag_all[i] == 3)
			{
				center_five.push_back(center_five_tmp[i]);
				resultVec_5.push_back(resultVec_5_tmp[i]);
				flag_d2 = true;
				d_num = i;
				break;
			}
		}
	}

	if ((flag_d1 == false) && (flag_d2 == false))
	{
		for (int i = 0; i < sortedRect.size(); i++)
		{
			if ((i == train_num) || (i == station1_num) || (i == station2_num))
				continue;
			if (info_d_flag_all[i] == 2)
			{
				center_five.push_back(center_five_tmp[i]);
				resultVec_5.push_back(resultVec_5_tmp[i]);
				flag_d3 = true;
				d_num = i;
				break;
			}
		}
	}

	if ((flag_d1 == false) && (flag_d2 == false) && (flag_d3 == false))
	{
		cout << "date can not found !!" << endl;
		return -1;
	}

	int sel_d = 0;
	if (flag_d1)
		sel_d = 1;
	else
	{
		if (flag_d2)
			sel_d = 0.5;
	}

	/*match position*/
	bool info_p_flag[50][4];
	int info_p_flag_all[50];
	for (int i = 0; i < sortedRect.size(); i++)
	{
		for (int j = 0; j < 4; j++)
		{
			info_p_flag[i][j] = false;
		}
	}
	bool flag_p1 = false;
	bool flag_p2 = false;
	bool flag_p3 = false;
	bool flag_p4 = false;
	int p_num;

	for (int i = 0; i < sortedRect.size(); i++)
	{
#if (debug1)
			printf("第%d个座位宽，高，中心横坐标差，中心纵坐标差是%f,%f,%f,%f\n", i, center_five_tmp[i][0], center_five_tmp[i][1], center_five_tmp[i][2] - center_five_tmp[train_num][2], center_five_tmp[i][3] - center_five_tmp[train_num][3]);
#endif
		if ((center_five_tmp[i][0] > 150) && (center_five_tmp[i][0] < 310))
			info_p_flag[i][0] = true;
		if ((center_five_tmp[i][1] > 30) && (center_five_tmp[i][1] < 50))
			info_p_flag[i][1] = true;
		if ((center_five_tmp[i][2] - center_five_tmp[train_num][2] > 170) && (center_five_tmp[i][2] - center_five_tmp[train_num][2]< 280))
			info_p_flag[i][2] = true;
		if ((center_five_tmp[i][3] - center_five_tmp[train_num][3]> 60) && (center_five_tmp[i][3] - center_five_tmp[train_num][3]< 120))
			info_p_flag[i][3] = true;

		info_p_flag_all[i] = int(info_p_flag[i][0]) + int(info_p_flag[i][1]) + int(info_p_flag[i][2]) + int(info_p_flag[i][3]);

#if (debug1)
			printf("第%d个座位标志是%d,%d,%d,%d\n", i, info_p_flag[i][0], info_p_flag[i][1], info_p_flag[i][2], info_p_flag[i][3]);
#endif
	}

	for (int i = 0; i < sortedRect.size(); i++)
	{
		if ((i == train_num) || (i == station1_num) || (i == station2_num) || (i == d_num))
			continue;
		if (info_p_flag_all[i] == 4)
		{
			center_five.push_back(center_five_tmp[i]);
			resultVec_5.push_back(resultVec_5_tmp[i]);
			flag_p1 = true;
			p_num = i;
			break;
		}
	}

	if (flag_p1 == false)
	{
		for (int i = 0; i < sortedRect.size(); i++)
		{
			if ((i == train_num) || (i == station1_num) || (i == station2_num) || (i == d_num))
				continue;
			if (info_p_flag_all[i] == 3)
			{
				center_five.push_back(center_five_tmp[i]);
				resultVec_5.push_back(resultVec_5_tmp[i]);
				flag_p2 = true;
				p_num = i;
				break;
			}
		}
	}

	if ((flag_p1 == false) && (flag_p2 == false))
	{
		for (int i = 0; i < sortedRect.size(); i++)
		{
			if ((i == train_num) || (i == station1_num) || (i == station2_num) || (i == d_num))
				continue;
			if (info_p_flag_all[i] == 2)
			{
				center_five.push_back(center_five_tmp[i]);
				resultVec_5.push_back(resultVec_5_tmp[i]);
				flag_p3 = true;
				p_num = i;
				break;
			}
		}
	}

	if ((flag_p1 == false) && (flag_p2 == false) && (flag_p3 == false))
	{
		cout << "position can not found !!" << endl;
		return -1;
	}

	int sel_p = 0;
	if (flag_p1)
		sel_p = 1;
	else
	{
		if (flag_p2)
			sel_p = 0.5;
	}

	if (((sel_t + sel_s1 + sel_s2 + sel_d + sel_p) < 3.5) && (line_threshold > 140))
	{
		cout << "text block all info selected is less than 4" << endl;
		return -1;
	}

	/*
	if (((sel_t + sel_s1 + sel_s2 + sel_d + sel_p) < 3.5))
	{
		cout << "text block all info selected is less than 4" << endl;
		return -1;
	}
	*/

	/*
	if (!((train_num < station1_num) && (station1_num < station2_num) && (station2_num < d_num) && (d_num < p_num)))
	{
		cout << "5 key infomation is not in order" << endl;
		return -1;
	}
	*/

	/*
	if (debug1)
	{
	printf("第%d张车票的贝叶斯训练信息如下：\n", picnum);
	printf("train num width,height,center.x,center.y is %f,%f,%f,%f\n", center_five[0][0], center_five[0][1], center_five[0][2], center_five[0][3]);
	for (int i = 1; i < 5; i++)
	{
	float x = center_five[i][2] - center_five[0][2];
	float y = center_five[i][3] - center_five[0][3];
	float width = center_five[i][0];
	float height = center_five[i][1];
	printf("from %d to info_t:width is %f,height is %f,(x,y) is %f,%f\n", i, width, height, x, y);
	}
	printf("\n");
	}
	*/

	/*
	if (debug1)
	{
	printf("\n");
	printf("train num is %d,station1 num is %d, station2 num is %d,date is %d,position is %d\n", train_num, station1_num, station2_num, d_num, p_num);
	printf("train num width,height,center.x,center.y is %f,%f,%f,%f\n", center_five[0][0], center_five[0][1], center_five[0][2], center_five[0][3]);
	printf("station 1 width,height,center.x,center.y is %f,%f,%f,%f\n", center_five[1][0], center_five[1][1], center_five[1][2], center_five[1][3]);
	printf("station 2 width,height,center.x,center.y is %f,%f,%f,%f\n", center_five[2][0], center_five[2][1], center_five[2][2], center_five[2][3]);
	printf("date width,height,center.x,center.y is %f,%f,%f,%f\n", center_five[3][0], center_five[3][1], center_five[3][2], center_five[3][3]);
	printf("position width,height,center.x,center.y is %f,%f,%f,%f\n", center_five[4][0], center_five[4][1], center_five[4][2], center_five[4][3]);
	}
	*/
	return 0;
}

/************************/
/*	 char selection		*/
/************************/
bool verifySizes_char(const Mat &auxRoi, const float &r, const float &area, const float &duty_cycle, const int &t, const int &j)
{
	if ((duty_cycle > duty_cycle_char) && (r > aspect_ratio_threshold_min_char) && (r < aspect_ratio_threshold_char_max) && (auxRoi.rows >= Height_min_char) && (auxRoi.rows <= Height_max_char) && (area > area_threshold_char))
	{
		return true;
	}
	else
		return false;
}

/********************************/
/*	   char preprocessing		*/
/********************************/
Mat preprocessChar(const Mat &in)
{
	//Remap image
	int h = in.rows;
	int w = in.cols;
	//int charSize = CHAR_SIZE;	//统一每个字符的大小
	Mat transformMat = Mat::eye(2, 3, CV_32F);
	int m = max(w, h);
	transformMat.at<float>(0, 2) = m / 2 - w / 2;
	transformMat.at<float>(1, 2) = m / 2 - h / 2;

	Mat warpImage(m, m, in.type());
	warpAffine(in, warpImage, transformMat, warpImage.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));

	return warpImage;
}

/************************/
/*	   sort char		*/
/************************/
int SortRect(const vector<Rect>& vecRect, vector<Rect>& out)
{
	vector<int> orderIndex;
	vector<int> xpositions;

	for (int i = 0; i < vecRect.size(); i++)
	{
		orderIndex.push_back(i);
		xpositions.push_back(vecRect[i].x);
	}

	float min = xpositions[0];
	int minIdx = 0;
	for (int i = 0; i< xpositions.size(); i++)
	{
		min = xpositions[i];
		minIdx = i;
		for (int j = i; j<xpositions.size(); j++)
		{
			if (xpositions[j]<min){
				min = xpositions[j];
				minIdx = j;
			}
		}
		int aux_i = orderIndex[i];
		int aux_min = orderIndex[minIdx];
		orderIndex[i] = aux_min;
		orderIndex[minIdx] = aux_i;

		float aux_xi = xpositions[i];
		float aux_xmin = xpositions[minIdx];
		xpositions[i] = aux_xmin;
		xpositions[minIdx] = aux_xi;
	}

	for (int i = 0; i<orderIndex.size(); i++)
	{
		out.push_back(vecRect[orderIndex[i]]);
	}

	return 0;
}

/************************/
/*	 char segmentation  */
/************************/
int charsSegment(Mat input, vector<Mat>& resultVec, const string &s1, const string &s2, const int &j) 
{
	if (!input.data)
	{
		cout << "none of picture inputed to process in charSegment funciton" << endl;
		return -1;
	}

#if (debug1)
		printf("\n开始字符分割\n");
#endif

	Mat input_clone = input.clone();
	string s_output;
	string s_close;
	string s_binary;
#if (debug2)
	{
		s_output = "F:\\图片\\车票图片\\处理后图片test\\b";
		s_output = s_output + s1 + s2 + ".jpg";
		s_close = "F:\\图片\\车票图片\\文字行区域闭操作\\b";
		s_close = s_close + s1 + s2 + ".jpg";
		s_binary = "F:\\图片\\车票图片\\文字行区域二值化\\b";
		s_binary = s_binary + s1 + s2 + ".jpg";
	}
#endif
	
	/*gray image*/
	cvtColor(input, input, CV_RGB2GRAY);

	/*binary image*/
	Mat img_threshold;
	threshold(input, img_threshold, 10, 255, CV_THRESH_OTSU + CV_THRESH_BINARY_INV);
	Mat img_threshold_clone = img_threshold.clone();
#if (debug2)
		imwrite(s_binary, img_threshold);
#endif

	/*close operation*/
	Mat element = getStructuringElement(MORPH_RECT, Size(MORPH_SIZE_WIDTH_CHAR, MORPH_SIZE_HEIGHT_CHAR));
	morphologyEx(img_threshold, img_threshold, CV_MOP_CLOSE, element);
#if (debug2)
		imwrite(s_close, img_threshold);
#endif

	/*extract contours*/
	Mat img_contours;
	img_threshold.copyTo(img_contours);

	vector< vector< Point> > contours;
	findContours(img_contours,
		contours, // a vector of contours
		CV_RETR_EXTERNAL, // retrieve the external contours
		CV_CHAIN_APPROX_NONE); // all pixels of each contours

	Mat contours_detect;
	Mat contours_detect_clone;
#if (debug2)
	{
		contours_detect = Mat::zeros(input.rows, input.cols, CV_8UC3);
		drawContours(contours_detect, contours, -1, Scalar(255, 255, 255), 1, 8);
		contours_detect.clone();
	}
#endif
		
	//Start to iterate to each contour founded
	vector<vector<Point> >::iterator itc = contours.begin();

	//Remove patch that are no inside limits of aspect ratio and area.
	int t = 0;
	vector<Rect> vecRect;
	while (itc != contours.end())
	{
		Rect mr = boundingRect(Mat(*itc));
#if (debug2)
		{
			rectangle(contours_detect, mr, Scalar(0, 0, 255));
			string words = to_string(t);
			putText(contours_detect, words, Point(mr.x, (mr.y + mr.height)), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(255, 255, 255));
			imwrite(s_output, contours_detect);
		}
#endif

		float contour_area = fabs(contourArea(*itc));
		float r = float(mr.width) / float(mr.height);
		int area = mr.width * mr.height;
		float duty_cycle = contour_area / area;

#if (debug1)			//no selection
			printf("第%d张文字区域图第%d个未筛选长宽比为%f，面积为%d,宽为%d，高为%d, 占空比为%f\n", j, t, r, area, mr.width, mr.height, duty_cycle);
#endif

		Mat auxRoi(img_threshold_clone, mr);
		if (verifySizes_char(auxRoi, r, area, duty_cycle, t, j))
		{
			vecRect.push_back(mr);
#if (debug1)
				printf("第%d张文字区域图第%d个筛选后的长宽比为%f，面积为%d,宽为%d，高为%d, 占空比为%f\n", j, t, r, area, mr.width, mr.height, duty_cycle);
#endif
		}
		++itc;
		t++;
	}

	if (vecRect.size() == 0)
	{
		cout << "there is no char after selection in" << j << " text block" << endl;
		return -1;
	}
		
	vector<Rect> sortedRect;
	SortRect(vecRect, sortedRect);
#if (debug1)
		printf("\ntest排序完成\n");
#endif

	for (int i = 0; i < sortedRect.size(); i++)
	{
		Rect mr = sortedRect[i];
		Mat auxRoi(img_threshold_clone, mr);

		auxRoi = preprocessChar(auxRoi);
		resultVec.push_back(auxRoi);
	}

	return 0;
}

/********************************/
/*	Perspective transformation 	*/
/********************************/
Mat PerspectiveTrans(const Mat &src, const Point2f* const scrPoints, const Point2f* const dstPoints)
{
	Mat dst;
	Mat Trans = getPerspectiveTransform(scrPoints, dstPoints);						//matrix3*3
	warpPerspective(src, dst, Trans, Size(src.cols, src.rows), CV_INTER_CUBIC);		//poerspective transformation
	return dst;
}

/************************/
/*	 image sharpen 		*/
/************************/
void sharpenImage1(const Mat &image, Mat &result)
{
	//create filter template
	Mat kernel(3, 3, CV_32F, cv::Scalar(0));
	kernel.at<float>(1, 1) = 5.0;
	kernel.at<float>(0, 1) = -1.0;
	kernel.at<float>(1, 0) = -1.0;
	kernel.at<float>(1, 2) = -1.0;
	kernel.at<float>(2, 1) = -1.0;

	result.create(image.size(), image.type());

	filter2D(image, result, image.depth(), kernel);					//对图像进行滤波(拉普拉斯算子锐化)
}

/************************/
/*	    quick sort 		*/
/************************/
void quick_sort(vector<Vec2f> &nums, const int &b, const int &e, const int &index)			//small to big
{
	if (b < e - 1)
	{
		int lb = b, rb = e - 1;
		while (lb < rb)
		{
			while (nums[rb][index] >= nums[b][index] && lb < rb)
				rb--;
			while (nums[lb][index] <= nums[b][index] && lb < rb)
				lb++;
			swap(nums[lb], nums[rb]);
		}
		swap(nums[b], nums[lb]);
		quick_sort(nums, b, lb, index);
		quick_sort(nums, lb + 1, e, index);
	}
}

/********************************************/
/*	   calculate line intersection point 	*/
/********************************************/
void calculateCorner(const float &rho1, const float &theta1, const float &rho2, const float &theta2, Point2f *point)
{
	if (theta1 != 0 && theta1 != CV_PI && theta2 != 0 && theta2 != CV_PI)
	{
		double a1 = cos(theta1), b1 = sin(theta1);
		double x01 = a1*rho1, y01 = b1*rho1;
		double x11 = cvRound(x01 + 1000 * (-b1));
		double y11 = cvRound(y01 + 1000 * (a1));
		double k1 = (y11 - y01) / (x11 - x01);

		double a2 = cos(theta2), b2 = sin(theta2);
		double x02 = a2*rho2, y02 = b2*rho2;
		double x12 = cvRound(x02 + 1000 * (-b2));
		double y12 = cvRound(y02 + 1000 * (a2));
		double k2 = (y12 - y02) / (x12 - x02);

		double x_tmp = (k1*x01 - k2*x02 + y02 - y01) / (k1 - k2);
		point->x = cvRound(x_tmp);
		point->y = cvRound(k1*(x_tmp - x01) + y01);
	}
	else
	{
		if (theta1 == 0 || theta1 == CV_PI)
		{
			double a1 = cos(theta1), b1 = sin(theta1);
			double x01 = a1*rho1, y01 = b1*rho1;

			double a2 = cos(theta2), b2 = sin(theta2);
			double x02 = a2*rho2, y02 = b2*rho2;
			double x12 = cvRound(x02 + 1000 * (-b2));
			double y12 = cvRound(y02 + 1000 * (a2));
			double k2 = (y12 - y02) / (x12 - x02);

			point->x = cvRound(x01);
			point->y = cvRound(k2*(x01 - x02) + y02);
		}
		else
		{
			double a1 = cos(theta1), b1 = sin(theta1);
			double x01 = a1*rho1, y01 = b1*rho1;
			double x11 = cvRound(x01 + 1000 * (-b1));
			double y11 = cvRound(y01 + 1000 * (a1));
			double k1 = (y11 - y01) / (x11 - x01);

			double a2 = cos(theta2), b2 = sin(theta2);
			double x02 = a2*rho2, y02 = b2*rho2;

			point->x = cvRound(x02);
			point->y = cvRound(k1*(x02 - x01) + y01);
		}
	}
}

/****************************/
/*	  ticket adjustment 	*/
/****************************/
bool Imageadjust(const Mat &srcImage1, Mat &result)
{
	Mat srcImage = srcImage1.clone();
	//guss filter
	Mat gaussianBlurImage;
	GaussianBlur(srcImage, gaussianBlurImage, Size(11, 11), 0, 0);
#if (debug_adjust)
		imshow("gaussianBlurImage", gaussianBlurImage);
#endif

	//sharpen
	Mat sharpenImage;
	sharpenImage1(gaussianBlurImage, sharpenImage);
#if (debug_adjust)
	{
		namedWindow("sharpenImage", WINDOW_NORMAL);
		resizeWindow("sharpenImage", srcImage.cols / 2, srcImage.rows / 2);
		imshow("sharpenImage", sharpenImage);
	}
#endif

	//gray image
	Mat srcGray;
	cvtColor(sharpenImage, srcGray, CV_RGB2GRAY);
#if (debug_adjust)
		imshow("srcGray", srcGray);
#endif

	//canny
	Mat midImage;
	Canny(srcGray, midImage, canny_low_threshold, canny_high_threshold, 3);			//low is 30，height is 200
#if (debug_adjust)
	{
		namedWindow("midImage", WINDOW_NORMAL);
		resizeWindow("midImage", srcImage.cols / 2, srcImage.rows / 2);
		imshow("midImage", midImage);
	}
#endif
	//hough line transform
	vector<Vec2f> lines;//定义一个矢量结构lines用于存放得到的线段矢量集合
	//HoughLines(midImage, lines, 1, CV_PI / 180, 150, 0, 0);
	HoughLines(midImage, lines, 1, CV_PI / 180, line_threshold, 0, 0);

	//sort by angle
#if (debug1)
		cout << "源数据变换后：" << endl;
#endif
	vector<Vec2f> linesTmp;
	for (int i = 0; i < lines.size(); i++)
	{

		linesTmp.push_back(lines[i]);
		if (lines[i][0] < 0)
		{
			linesTmp[i][0] = abs(linesTmp[i][0]);  //(将距离转化为正，负距离下的角度变为theta-PI)（能够处理梯形变形，不能处理特殊平行四边形（平行四边形有两条边分别在原点两侧））
			linesTmp[i][1] = linesTmp[i][1] - CV_PI;
		}
#if (debug1)
		{
			cout << "linesTmp[" << i << "][0]:" << linesTmp[i][0] << endl;
			cout << "linesTmp[" << i << "][1]:" << linesTmp[i][1] << endl;
		}
#endif
	}

#if (debug1)
		cout << "按角度排序（由小到大）：" << endl;
#endif
	quick_sort(linesTmp, 0, linesTmp.size(), 1);

#if (debug1)
	{
		for (int i = 0; i < linesTmp.size(); i++)
		{
			cout << "linesTmp[" << i << "][0]:" << linesTmp[i][0] << endl;
			cout << "linesTmp[" << i << "][1]:" << linesTmp[i][1] << endl;
		}
	}
#endif

	cout << endl;
	bool index_flag = true;
	vector<Vec2f> index;
	Vec2f indexTmp;
	for (int i = 0; i < linesTmp.size(); i++)
	{
		while ((i < linesTmp.size()) && (linesTmp[i][1] > -0.14) && (linesTmp[i][1] < 0.14))
		{
			if (index_flag)
			{
				indexTmp[0] = i;
				index_flag = false;
			}
			i++;
		}
		indexTmp[1] = i;
		if (!index_flag)
			break;
	}

	if (index_flag)
	{
		cout << "Error, The short edge near the origin is not found!!" << endl;
#if (debug6)
		return false;
#endif
	}

	index.push_back(indexTmp);
	if (index[0][0] == index[0][1])
	{
		cout << "Error, The short edge near the origin is the same as the short edge far from the origin!!";
#if (debug6)
		return false;
#endif
	}
	index_flag = true;

	for (int i = index[0][1]; i < linesTmp.size(); i++)
	{
		while ((i < linesTmp.size()) && (linesTmp[i][1] > 1.43) && (linesTmp[i][1] < 1.71))
		{
			if (index_flag)
			{
				indexTmp[0] = i;
				index_flag = false;
			}
			i++;
		}
		indexTmp[1] = i;
		if (!index_flag)
			break;
	}

	if (index_flag)
	{
		cout << "Error, The long edge near the origin is not found!!" << endl;
#if (debug6)
		return false;
#endif
	}

	index.push_back(indexTmp);
	if (index[1][0] == index[1][1])
	{
		cout << "Error, The long edge near the origin is the same as the long edge far from the origin!!";
#if (debug6)
		return false;
#endif
	}

	if (index.size() <= 1)
	{
		cout << "Error, only one edge ji detected!!";
#if (debug6)
		return false;
#endif
	}

#if (debug1)
	{
		for (int i = 0; i < index.size(); i++)
		{
			cout << "index[" << i << "]:" << index[i][0] << " to " << index[i][1] - 1 << endl;
		}
	}
#endif

	//sort by rho
#if (debug1)
		cout << "分组后，每组按距离排序（由小到大）：" << endl;
#endif
	for (int i = 0; i < index.size(); i++)
	{
		quick_sort(linesTmp, index[i][0], index[i][1], 0);
	}
#if (debug1)
	{
		for (int i = index[0][0]; i < index[0][1]; i++)
		{
			cout << "linesTmp[" << i << "][0]:" << linesTmp[i][0] << endl;
			cout << "linesTmp[" << i << "][1]:" << linesTmp[i][1] << endl;
		}
	}
#endif
#if (debug1)
	{
		for (int i = index[1][0]; i < index[1][1]; i++)
		{
			cout << "linesTmp[" << i << "][0]:" << linesTmp[i][0] << endl;
			cout << "linesTmp[" << i << "][1]:" << linesTmp[i][1] << endl;
		}
		cout << endl;
	}
#endif
		
	int indexMaxBeg1 = index[0][0];			//ticket shorter size
	int indexMaxEnd1 = index[0][1];
	int indexMaxBeg2 = index[1][0];			//ticket longer size
	int indexMaxEnd2 = index[1][1];


	//draw line
	Mat srcImage_clone = srcImage.clone();
	Mat srcImage_clone2 = srcImage.clone();

#if (debug2)
	{
		for (size_t i = 0; i < lines.size(); i++)
		{
			float rho = linesTmp[i][0], theta = linesTmp[i][1];

			if (i == indexMaxBeg1 || i == indexMaxEnd1 - 1 || i == indexMaxBeg2 || i == indexMaxEnd2 - 1)
			{
				Point pt1, pt2;
				double a = cos(theta), b = sin(theta);
				double x0 = a*rho, y0 = b*rho;
				pt1.x = cvRound(x0 + 1000 * (-b));
				pt1.y = cvRound(y0 + 1000 * (a));
				pt2.x = cvRound(x0 - 1000 * (-b));
				pt2.y = cvRound(y0 - 1000 * (a));
				line(srcImage, pt1, pt2, Scalar(0, 0, 255), 1, CV_AA);
			}
		}

		namedWindow("待检测目标的四条边", WINDOW_NORMAL);
		resizeWindow("待检测目标的四条边", srcImage.cols / 2, srcImage.rows / 2);
		imshow("待检测目标的四条边", srcImage);

		for (size_t i = 0; i < lines.size(); i++)
		{
			float rho = linesTmp[i][0], theta = linesTmp[i][1];

			Point pt1, pt2;
			double a = cos(theta), b = sin(theta);
			double x0 = a*rho, y0 = b*rho;
			pt1.x = cvRound(x0 + 1000 * (-b));
			pt1.y = cvRound(y0 + 1000 * (a));
			pt2.x = cvRound(x0 - 1000 * (-b));
			pt2.y = cvRound(y0 - 1000 * (a));
			line(srcImage_clone, pt1, pt2, Scalar(0, 0, 255), 1, CV_AA);
		}

		namedWindow("待检测目标的所有边", WINDOW_NORMAL);
		resizeWindow("待检测目标的所有边", srcImage.cols / 2, srcImage.rows / 2);
		imshow("待检测目标的所有边", srcImage_clone);
	}
#endif

	Point2f AffinePoints0[4];
	Point2f AffinePoints1[4];
	Point2f pointTmp;
	calculateCorner(linesTmp[indexMaxBeg1][0], linesTmp[indexMaxBeg1][1], linesTmp[indexMaxBeg2][0], linesTmp[indexMaxBeg2][1], &pointTmp);
	//circle(srcImage, pointTmp, 2, Scalar(0, 0, 255), 2);
	AffinePoints0[0] = pointTmp;
	calculateCorner(linesTmp[indexMaxBeg1][0], linesTmp[indexMaxBeg1][1], linesTmp[indexMaxEnd2 - 1][0], linesTmp[indexMaxEnd2 - 1][1], &pointTmp);
	//circle(srcImage, pointTmp, 2, Scalar(0, 255, 0), 2);
	AffinePoints0[1] = pointTmp;
	calculateCorner(linesTmp[indexMaxEnd1 - 1][0], linesTmp[indexMaxEnd1 - 1][1], linesTmp[indexMaxBeg2][0], linesTmp[indexMaxBeg2][1], &pointTmp);
	//circle(srcImage, pointTmp, 2, Scalar(255, 0, 0), 2);
	AffinePoints0[2] = pointTmp;
	calculateCorner(linesTmp[indexMaxEnd1 - 1][0], linesTmp[indexMaxEnd1 - 1][1], linesTmp[indexMaxEnd2 - 1][0], linesTmp[indexMaxEnd2 - 1][1], &pointTmp);
	//circle(srcImage, pointTmp, 2, Scalar(0, 0, 255), 2);
	AffinePoints0[3] = pointTmp;

	//judge if longer size if longer than shorter size
	int longer_x = AffinePoints0[2].x + AffinePoints0[3].x - AffinePoints0[0].x - AffinePoints0[1].x;
	int shorter_y = AffinePoints0[1].y + AffinePoints0[3].y - AffinePoints0[0].y - AffinePoints0[2].y;

	if (longer_x < shorter_y)
	{
		cout << "shorter size is longer than longer size" << endl;
		return false;
	}

	AffinePoints1[0] = Point2f(200, 200);
	AffinePoints1[1] = Point2f(200, 800);
	AffinePoints1[2] = Point2f(1100, 200);
	AffinePoints1[3] = Point2f(1100, 800);

	Mat dst_perspective = PerspectiveTrans(srcImage_clone2, AffinePoints0, AffinePoints1);
	result = dst_perspective.clone();

	//draw four spot
#if (debug2)
	{
		for (int i = 0; i < 4; i++)
		{
			//circle(srcImage, AffinePoints0[i], 2, Scalar(0, 0, 255), 2);
			circle(srcImage, AffinePoints0[i], 5, Scalar(0, 255, 0), 5);				//校正前的点
			circle(dst_perspective, AffinePoints1[i], 2, Scalar(0, 0, 255), 2);				//校正后的点
		}

		//显示源图和目标图
		namedWindow("originpoint", WINDOW_NORMAL);
		resizeWindow("originpoint", srcImage.cols / 2, srcImage.rows / 2);
		imshow("originpoint", srcImage);
		imshow("perspective", dst_perspective);
	}
#endif

	return true;
}

/****************************/
/*	  ticket detection 		*/
/****************************/
int ticket_detection(const string &s_input, output &ticket_output)
{
#if (debug3)
	time0 = (double)getTickCount();
#endif 

	//bayes model
	if (!load_bayes_flag)
	{
		normalBayes_capital_t.load("normalBayes_capital_t.txt");
		normalBayes_num_t.load("normalBayes_num_t.txt");
		normalBayes_char_s.load("normalBayes_char_s.txt");
		normalBayes_num_d.load("normalBayes_num_d.txt");
		normalBayes_lowercase_p.load("normalBayes_lowercase_p.txt");
		normalBayes_num_p.load("normalBayes_num_p.txt");
		load_bayes_flag = true;
	}
	
	//name of output picture
	string s_output;
	string s_output5;
	string s_output_char;
	string s_output_picadjust;
	string s_output_picadjust1;
	string s_output_char_feature;

#if (debug2)
	{
		s_output = "F:\\图片\\车票图片\\处理后图片\\b";
		s_output5 = "F:\\图片\\车票图片\\关键信息5\\b";
		s_output_char = "F:\\图片\\车票图片\\分割后的文字\\b";
		s_output_picadjust = "F:\\图片\\车票图片\\校正后的图片\\b";
		s_output_picadjust1 = "F:\\图片\\车票图片\\校正后的图片1\\b";
		s_output_char_feature = "F:\\图片\\车票图片\\分割后的文字切边\\b";
	}
#endif
	
	string s1;
	string s2;
	string s3;

#if (debug1)
		cout << "test_plate_locate" << endl;
#endif

	if (1)
	{
#if (debug5)
		{
			time4 = (double)getTickCount();
		}
#endif
		int picnum = count_pic;
		s1 = to_string(picnum);
		if (picnum < 10)
			s1 = "0" + s1;

#if (debug2)
		{
			s_output_picadjust = s_output_picadjust + s1 + ".jpg";
			s_output_picadjust1 = s_output_picadjust1 + s1 + ".jpg";
		}
#endif

		Mat srcImage = imread(s_input);
		if (srcImage.empty())
			return -1;
		while ((srcImage.rows < 800) || (srcImage.cols < 1100))			//pyramid up the ticket picture
		{
			pyrUp(srcImage, srcImage, Size(srcImage.cols * 2, srcImage.rows * 2));
			imwrite("pyrUp.jpg", srcImage);
		}

#if (debug2)
		imshow("Src", srcImage);
#endif

		/*control hough line threshold change or not*/
		bool hough_correct_flag = false;

		line_threshold = hough_line_threshold;
		while (!hough_correct_flag)
		{
			if (line_threshold < (hough_line_threshold - 10 * hough_line_dec))			//140
			{
				return -1;
			}

			/*ticket adjustment*/
			Mat img_adjust;
			//Mat img_adjust_sub;
			Mat img_adjust_sub_resize;
			//img_adjust_sub_resize.create(600, 900, srcImage.type());
			bool adjust_flag = Imageadjust(srcImage, img_adjust);
			if (!adjust_flag)
			{
				cout << "Error, picture adjustment is wrong!!" << endl;
				hough_correct_flag = false;
				line_threshold -= 10;
				continue;
			}

#if (debug2)
			imshow("图像预校正效果图", img_adjust);
#endif
			getRectSubPix(img_adjust, Size(900, 600), Point2f(650, 500), img_adjust_sub_resize);
#if (debug1)
			printf("\n图像的宽是%d和高是%d\n", srcImage.cols, srcImage.rows);
#endif
			//resize(img_adjust_sub, img_adjust_sub_resize, img_adjust_sub_resize.size(), 0, 0, INTER_CUBIC);			//要按照拍摄图片的大小来resize
#if (debug1)
			cout << endl << s_output_picadjust << endl;
#endif

#if (debug2)
			{
				imwrite(s_output_picadjust, img_adjust_sub_resize);
				imshow("预校正后的图像", img_adjust_sub_resize);
			}
#endif

#if (debug3)
			{
				cout << "ticket adjustment time is ";
				cout << ((double)getTickCount() - time0) / (double)getTickFrequency() << endl;
				time0 = (double)getTickCount();
			}
#endif

			/*text block segment*/
			vector<Mat> resultVec;
			vector<Mat> resultVec_5;
			Mat img_adjust_sub_resize1;
			int result = text_block_segment(img_adjust_sub_resize, resultVec, img_adjust_sub_resize1, resultVec_5, picnum);
			if (result == 0)
			{
				/*debug for 5 key information*/
#if (debug2)
				{
					imwrite(s_output_picadjust1, img_adjust_sub_resize1);		//show picture adjusted with rectangle
					for (int j = 0; j < resultVec_5.size(); j++)
					{
						s2 = to_string(j);
						if (j < 10)
							s2 = "0" + s2;
						s_output5 = s_output5 + s1 + s2 + ".jpg";
						imwrite(s_output5, resultVec_5[j]);
						s_output5 = "F:\\图片\\车票图片\\关键信息5\\b";
						s2 = "";
					}

					int num = resultVec.size();
					printf("检测第%d张车票得到%d个分割区域\n", picnum, num);					//在屏幕上打印输出信息，用于调试
					for (int j = 0; j < num; j++)
					{
						s2 = to_string(j);
						if (j < 10)
							s2 = "0" + s2;
						s_output = s_output + s1 + s2 + ".jpg";
						Mat resultMat = resultVec[j];
						imwrite(s_output, resultMat);
						printf("得到第%d张车票的第%d个分割区域\n", picnum, j);
						s_output = "F:\\图片\\车票图片\\处理后图片\\b";
						s2 = "";
					}
				}
#endif
			}
			else
			{
				cout << "Error, text block segment is wrong!!" << endl;
				hough_correct_flag = false;
				line_threshold -= 10;
				continue;
			}

#if (debug3)
			{
				cout << "text block segment time is ";
				cout << ((double)getTickCount() - time0) / (double)getTickFrequency() << endl;
				time0 = (double)getTickCount();
			}
#endif

			/*char segment*/
			vector<Mat> char_resultVec;
#if (debug3)
			time1 = (double)getTickCount();
#endif
			for (int j = 0; j < resultVec_5.size(); j++)
			{
				/*debug for each key information*/
				//if ((j != 0) && (j != 1) && (j != 2) && (j != 3))
				//if (j != 4)
				//if((j != 1) && (j != 2))
					//continue;

				Mat resultMat = resultVec_5[j];				//5 key information
				string plateIdentify = "";					//print structured information
				s2 = to_string(j);
				if (j < 10)
					s2 = "0" + s2;
				vector<Mat> char_resultVec;					//char vector after char segment
				int char_result = charsSegment(resultMat, char_resultVec, s1, s2, j);
				Mat szx;
				if (j == 4)
				{
					if (char_resultVec.size() > 1)
						szx = char_resultVec[char_resultVec.size() - 2];
					else
						szx = char_resultVec[0];
				}
					
#if (debug3)
				{
					cout << "\t each five text block's char segment time is ";
					cout << ((double)getTickCount() - time1) / (double)getTickFrequency() << endl;
					time1 = (double)getTickCount();
				}
#endif
				int p_sum = 0;
				if (char_result == 0)
				{
					int num = char_resultVec.size();

					/*
					if ((j == 0) && (num < 4))
					{
						cout << "train num is less than 4" << endl;
						return -1;
					}
					*/

					for (int k = 0; k < num; k++)
					{
						Mat char_resultMat = char_resultVec[k];
#if (debug2)
						{
							s3 = to_string(k);
							if (k < 10)
								s3 = "0" + s3;
							s_output_char = s_output_char + s1 + s2 + "_" + s3 + ".bmp";
							s_output_char_feature = s_output_char_feature + s1 + s2 + "_" + s3;
							imwrite(s_output_char, char_resultMat);
							s_output_char = "F:\\图片\\车票图片\\分割后的文字\\b";
						}
#endif

						/*char recognition*/
						bool isChinses = false;
						vector<Mat> qie_bian_pic;
						string charcater;
						if (j == 0)
						{
							if (k == 0)
							{
								charcater = charsIdentify(char_resultMat, qie_bian_pic, j, k, szx, num, p_sum, normalBayes_capital_t, normalBayes_lowercase_p);
							}
							else
							{
								charcater = charsIdentify(char_resultMat, qie_bian_pic, j, k, szx, num, p_sum, normalBayes_num_t, normalBayes_lowercase_p);
							}
						}

						if ((j == 1) || (j == 2))
						{
							charcater = charsIdentify(char_resultMat, qie_bian_pic, j, k, szx, num, p_sum, normalBayes_char_s, normalBayes_lowercase_p);
						}

						if (j == 3)
						{
							charcater = charsIdentify(char_resultMat, qie_bian_pic, j, k, szx, num, p_sum, normalBayes_num_d, normalBayes_lowercase_p);
						}

						if (j == 4)
						{
							charcater = charsIdentify(char_resultMat, qie_bian_pic, j, k, szx, num, p_sum, normalBayes_num_p, normalBayes_lowercase_p);
						}

						/*char re-recognition*/
						int len = charcater.size();
						for (int i = 0; i < len; i++)
						{
							if (charcater[i] == '*')
							{
								//charcater[i] = '\0';
								charcater.erase(i, 1);
								len = charcater.size();
								i--;
							}
						}

						if (charcater == "(")
						{
							charcater = "0";
						}
						if (charcater == ")")
						{
							charcater = "";
						}


						plateIdentify = plateIdentify + charcater;
#if (debug3)
						{
							cout << "\t each char recognition time is ";
							cout << ((double)getTickCount() - time1) / (double)getTickFrequency() << endl;
							time1 = (double)getTickCount();
						}
#endif
#if (debug2)
						{
							for (int l = 0; l < qie_bian_pic.size(); l++)
							{
								string s4 = to_string(l);
								s_output_char_feature = s_output_char_feature + "_" + s4 + ".bmp";
								imwrite(s_output_char_feature, qie_bian_pic[l]);
								s_output_char_feature = "F:\\图片\\车票图片\\分割后的文字切边\\b" + s1 + s2 + "_" + s3;
							}
							s3 = "";
							s_output_char_feature = "F:\\图片\\车票图片\\分割后的文字切边\\b";
						}
#endif
					}
				}
				else
				{
					cout << "Error, char segment is wrong!!";
					hough_correct_flag = false;
					line_threshold -= 10;
					continue;
				}

				/*write structured information to output sturct*/
				if (j == 0)
				{
					ticket_output.info_t = plateIdentify;
				}
				if (j == 1)
				{
					ticket_output.info_s1 = plateIdentify;
				}
				if (j == 2)
				{
					ticket_output.info_s2 = plateIdentify;
				}
				if (j == 3)
				{
					ticket_output.info_d = plateIdentify;
				}
				if (j == 4)
				{
					ticket_output.info_p = plateIdentify;
				}
				s2 = "";
#if (debug3)
				{
					cout << "\t write to txt time is ";
					cout << ((double)getTickCount() - time1) / (double)getTickFrequency() << endl;
					time1 = (double)getTickCount();
				}
#endif
			}
#if (debug3)
			{
				cout << "char segment and recognition time is ";
				cout << ((double)getTickCount() - time0) / (double)getTickFrequency() << endl;
				time0 = (double)getTickCount();
			}
#endif
			hough_correct_flag = true;
#if (debug5)
			cout << endl << "hough line threshold is " << line_threshold << endl;
#endif
		}

#if (debug2)
		{
			s_output_picadjust = "F:\\图片\\车票图片\\校正后的图片\\b";
			s_output_picadjust1 = "F:\\图片\\车票图片\\校正后的图片1\\b";
		}
#endif
		s1 = "";
#if (debug3)
		{
			cout << "left time is ";
			cout << ((double)getTickCount() - time0) / (double)getTickFrequency() << endl;
			time0 = (double)getTickCount();
		}
#endif
#if (debug5)
		{
			cout << "each ticket process time is ";
			cout << ((double)getTickCount() - time4) / (double)getTickFrequency() << endl;
			time4 = (double)getTickCount();
		}
#endif
	}




	return 0;
}

/************************/
/*	  main function    	*/
/************************/
int main()
{
	/*picture name*/
	//string s_input = "1.jpg";
	string s_input = "待处理图片\\b";
	string s1;
	/*output txt file*/
	ofstream outfile;
	outfile.open("train.txt", ios::out);
	output ticket_output;
	for (int picnum = 1; picnum <= 1; picnum++)
	{
#if (debug5)
			cout << "开始处理第 " << picnum << "张车票" << endl;
#endif
		/*debug with one picture*/
		//if (picnum != selective_picture)
		//continue;

		/*picture 11 and 18 is incorrect*/
		if ((picnum == 11) || (picnum == 18))
			continue;
		s1 = to_string(picnum);
		if (picnum < 10)
			s1 = "0" + s1;
		s_input = s_input + s1 + ".jpg";
		/*注意，因为训练的时候是用.jpg的格式图片进行训练，所以输入图片务必先将摄像头图片imwrite成.jpg格式后再imread进去，不能直接读取摄像头图片*/
		count_pic = picnum;
		int reprocess_flag = ticket_detection(s_input, ticket_output);			
		if (reprocess_flag == -1)	/*please input one more picture from vedio*/						
		{
			cout << "ticket detection is wrong, please input one more picture to process!!" << endl;
			s_input = "待处理图片\\b";
			s1 = "";
			continue;
		}
#if (debug5)
		{
			cout << ticket_output.info_t << endl;
			cout << ticket_output.info_s1 << endl;
			cout << ticket_output.info_s2 << endl;
			cout << ticket_output.info_d << endl;
			cout << ticket_output.info_p << endl;
			cout << endl; 
		}
#endif
		outfile << "第" << picnum << "张车票的列车号信息检测结果为：" << ticket_output.info_t << endl;
		outfile << "第" << picnum << "张车票的起始站名信息检测结果为：" << ticket_output.info_s1 << endl;
		outfile << "第" << picnum << "张车票的终止站名信息检测结果为：" << ticket_output.info_s2 << endl;
		outfile << "第" << picnum << "张车票的日期信息检测结果为：" << ticket_output.info_d << endl;
		outfile << "第" << picnum << "张车票的座位信息检测结果为：" << ticket_output.info_p << endl;
		outfile << endl;
		s_input = "待处理图片\\b";	
		s1 = "";
	}	
	outfile.close();
#if (debug5)
		cout << endl << "ticket detection is all over!!" << endl;
#endif
	waitKey();
	return 0;
}
