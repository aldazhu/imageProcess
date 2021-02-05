#pragma once
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2\opencv.hpp>

#ifndef IMAGEPROCESS_H
#define IMAGEPROCESS_H

#include <string>

int showImage(cv::Mat img, const std::string name, int waitMode = 1, int destroyWindowMode = 0);


cv::Mat sobel(cv::Mat grayImage, int ksize = 3);


cv::Mat gauss(cv::Mat img, int ksize = 3, double sigmax=0, double sigmay=0);

cv::Mat thresh(cv::Mat grayImage, double thresh, double maxval=255, int threshMode = 3);



cv::Mat adaptiveThreshold(cv::Mat grayImage, 
							double maxval = 255,
							int adaptiveMethod = cv::ADAPTIVE_THRESH_MEAN_C,
							int thresholdType = cv::THRESH_BINARY_INV, 
							int blockSize = 11,
							double C = 15);

cv::Mat otsu(cv::Mat grayImage, double maxval = 255);


cv::Mat open(cv::Mat binaryImage, int ksize=3, int kernelMode = 0);


cv::Mat close(cv::Mat binaryImage, int ksize = 3, int kernelMode = 2);


cv::Mat erode(cv::Mat binaryImage, int ksize = 3, int kernelMode = 0);

cv::Mat dilate(cv::Mat binaryImage, int ksize = 3, int kernelMode = 0);

std::vector<std::vector<cv::Point>> findContours(cv::Mat binaryImage, 
	int topologyMode = 1, int contoursType = 1);

cv::Mat drawContours(cv::Mat img, std::vector<std::vector<cv::Point>> contours,
	int contoursIdx = -1, int b=0, int g=0, int r=255);


cv::Mat watersherd(cv::Mat colorImage);


cv::Mat bgr2gray(cv::Mat img);


int accessPixels(cv::Mat img);


cv::Mat bilateralFilter(cv::Mat srcImage, int d=1, 
						           double sigmaColor=1, 
						double sigmaSpace=1);

#endif