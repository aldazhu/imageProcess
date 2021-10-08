#ifndef _IMAGE_PROCESS_H_
#define _IMAGE_PROCESS_H_

#include "opencv.hpp"

namespace cvbag
{
	bool isImageEmpty(const cv::Mat &image, const std::string functionName);

	int getAllImagePath(std::string folder, std::vector<cv::String> &imagePathList, bool flg = false);

	int showImage(const cv::Mat &image, const std::string winName = "img", const int waitKeyMode = 0, const int destroyMode = 0);

	int gaussBlur(const cv::Mat &image, cv::Mat &dst, const int ksize = 5, double sigma = 1.0);

	int sobel_x(const cv::Mat &image, cv::Mat &dst, const int ksize = 3);

	int sobel_y(const cv::Mat &image, cv::Mat &dst, const int ksize = 3);

	int sobel_xy(const cv::Mat &image, cv::Mat &dst, const int ksize = 3);

	int canny(const cv::Mat &image, cv::Mat &dst, const int low = 100, const int heigh = 200);

	int otsu(const cv::Mat &image, cv::Mat &dst);

	int threshold(const cv::Mat &image, cv::Mat &dst, const int th = 128, const int mode = 0, const int maxval = 255);

	int adaptiveThreshold(const cv::Mat &image, cv::Mat &dst, int blockSize = 11, double C = 15, double maxval = 255,
		int adaptiveMethod = cv::ADAPTIVE_THRESH_MEAN_C, int thresholdType = cv::THRESH_BINARY_INV);

	int  findContours(const cv::Mat &binaryImage, std::vector<std::vector<cv::Point>> &contours, int topologyMode = 1, int contoursType = 1);

	int drawContours(cv::Mat &image, const std::vector<std::vector<cv::Point>> &contours, int contoursIdx = -1, int b = 0, int g = 0, int r = 255);

	int dilate(const cv::Mat &binaryImage, cv::Mat &dst, const int ksize = 3, const int kernelMode = 0);

	int erode(const cv::Mat &binaryImage, cv::Mat &dst, const int ksize = 3, const int kernelMode = 0);

	int open(const cv::Mat &binaryImage, cv::Mat &dst, const int ksize = 3, const int kernelMode = 0);

	int close(const cv::Mat &binaryImage, cv::Mat &dst, const int ksize = 3, const int kernelMode = 0);

	//rotateImage()
	//angle: degree format ,eg: 10°,30°
	//mode = 0 ,Keep the original image size unchanged
	//mode = 1, Change the original image size to fit the rotated scale, padding with zero
	int rotateImage(const cv::Mat &srcImage, cv::Mat &outImage, const double angle, const int mode=1);
                  
	namespace match {
		//cpuTemplateMatch
		int cpuTemplateMatch(const cv::Mat &srcImage, const cv::Mat &tempImage, cv::Mat &result,
			double &matchVal, cv::Point &matchLoc, int mode);

		//Template Match With Angle
		int cpuTemplateMatchWithAngle(const cv::Mat &srcImage, const cv::Mat &tempImage, cv::Mat &result,
			double &matchVal, cv::Point &matchLoc, int mode, double &resultAngle,
			double angleStart = -10, double angleEnd = 10, double angleStep = 1);

		int fitCurve_2grade(std::vector<double> x, std::vector<double> y, double &finaly, double &finalx);

	}//end namespace match
	

	//use the table to transform the pixels 
	int gammaTransform(const cv::Mat &image, cv::Mat &dst, const int  table[]);
	//if the input image's format  is 3 channels then use below method to transform the pixels
	int gammaTransform_threeChannels(const cv::Mat &image, cv::Mat &dst, const int table[]);
	//a[i] = int(pow(i / 255.0, gamma) *255.0);
	int getGammaTable(const double gamma, int *a, const int num = 256);
	//gamma API
	int gamma(const cv::Mat &image, cv::Mat &dst, const double gamma = 1.0);

	//gamma piecewise linear function transform
	int getGammaTable_piecewiseLinear(int *a, const int src1, const int dst1, const int src2, const int dst2);
	//gamma_picewiseLinear 
	/*
		f(x) = (dst1 / src1) * x; if x<src1
		  = [(dst2 - dst1) / (src2 - src1)] * (x - src1) + dst1; if x>=src1 and x<src2;
		  = [(255 - dst2) / (255 - src2)] * (x - src2) + dst2; if x>=src2;
	*/
	int gamma_picewiseLinear(const cv::Mat &image, cv::Mat &dst,
		const int src1 = 80, const int dst1 = 60, const int src2 = 160, const int dst2 = 180);

}//end namespace cvbag

#endif