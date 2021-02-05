#pragma once
#include <string>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2\opencv.hpp>

#include "imageProcess.h"//my opencv function API


//////////////////////////////////////////////////////////////////
///input: 
///img			image cv::Mat
///name			windowName
///waitMode		waitMode=0 ==> cv::waitKey(0); waitMode=1 ==> cv::waitKey(1);
///destroyWindowMode	destroyWindowMode=0 ==>do not destroy window
///output:		-1 image is empty; 0 show image success;
//////////////////////////////////////////////////////////////////
int showImage(cv::Mat img, const std::string name, int waitMode = 1, int destroyWindowMode = 0)
{
	if (img.empty())
	{
		std::cout << "the image is empty!\n";
			return -1;
	}
	cv::namedWindow(name, 0);

	if (waitMode == 0 || waitMode == 1)
	{
		cv::imshow(name, img);
		cv::waitKey(waitMode);
	}
	else
	{
		cv::imshow(name, img);
		cv::waitKey(1);
	}

	if (destroyWindowMode == 1)
	{
		cv::destroyWindow(name);
	}

	return 0;
}

cv::Mat sobel(cv::Mat grayImage, int ksize = 3)
{
	/*
	cv::Sobel(
	InputArray Src,  //输入图像
	OutputArray dst,  //输出图像，大小与输入图像一致
	int depth,      //输出图像深度
	int dx,        //x方向，几阶导数
	int dy,        //y方向，几阶导数
	int ksize,      //Sobel算子的kernel大小，必须是1,3,5,7
	double scale=1,
	double delta=0,
	int borderType =BORDER_DEFAULT
	)
	*/

	if (grayImage.channels() == 3)
	{
		cv::cvtColor(grayImage, grayImage, cv::COLOR_BGR2GRAY);
	}

	cv::Mat xgrad;
	cv::Mat ygrad;
	cv::Mat xygrad;

	cv::Sobel(grayImage, xgrad, CV_16S, 1, 0, ksize);
	cv::Sobel(grayImage, ygrad, CV_16S, 0, 1, ksize);
	cv::convertScaleAbs(xgrad, xgrad);
	cv::convertScaleAbs(ygrad, ygrad);
	cv::addWeighted(xgrad, 0.5, ygrad, 0.5, 0, xygrad);

	return xygrad;
}

cv::Mat canny(cv::Mat &img, double threshold1 = 80, double threshold2 = 180)
{
	if (img.empty())
	{
		std::cout << "ERROR:In function poleDetect: the image is empty!\n";
		return img;
	}
	cv::Mat canny;
	cv::Canny(img, canny, threshold1, threshold2);//canny 8bits image
	//cv::imshow("canny", canny);
	//cv::waitKey(0);
	
	return canny;
}

cv::Mat gauss(cv::Mat img, int ksize = 3, double sigmax=0, double sigmay=0)
{
	if (img.empty()) return img;
	cv::GaussianBlur(img, img, cv::Size(ksize, ksize), sigmax, sigmay);
	return img;
}

cv::Mat thresh(cv::Mat grayImage, double thresh, double maxval=255, int threshMode = 3)
{
	/*
	enum ThresholdTypes {
    THRESH_BINARY     = 0, //!< \f[\texttt{dst} (x,y) =  \fork{\texttt{maxval}}{if \(\texttt{src}(x,y) > \texttt{thresh}\)}{0}{otherwise}\f]
    THRESH_BINARY_INV = 1, //!< \f[\texttt{dst} (x,y) =  \fork{0}{if \(\texttt{src}(x,y) > \texttt{thresh}\)}{\texttt{maxval}}{otherwise}\f]
    THRESH_TRUNC      = 2, //!< \f[\texttt{dst} (x,y) =  \fork{\texttt{threshold}}{if \(\texttt{src}(x,y) > \texttt{thresh}\)}{\texttt{src}(x,y)}{otherwise}\f]
    THRESH_TOZERO     = 3, //!< \f[\texttt{dst} (x,y) =  \fork{\texttt{src}(x,y)}{if \(\texttt{src}(x,y) > \texttt{thresh}\)}{0}{otherwise}\f]
    THRESH_TOZERO_INV = 4, //!< \f[\texttt{dst} (x,y) =  \fork{0}{if \(\texttt{src}(x,y) > \texttt{thresh}\)}{\texttt{src}(x,y)}{otherwise}\f]
    THRESH_MASK       = 7,
    THRESH_OTSU       = 8, //!< flag, use Otsu algorithm to choose the optimal threshold value
    THRESH_TRIANGLE   = 16 //!< flag, use Triangle algorithm to choose the optimal threshold value
};
	*/

	if (grayImage.empty()) return grayImage;
	if (grayImage.channels() == 3) cv::cvtColor(grayImage, grayImage, cv::COLOR_BGR2GRAY);
	cv::threshold(grayImage, grayImage, thresh, maxval,threshMode);

	return grayImage;
}

cv::Mat adaptiveThreshold(cv::Mat grayImage, 
							double maxval = 255,
							int adaptiveMethod = cv::ADAPTIVE_THRESH_MEAN_C,
							int thresholdType = cv::THRESH_BINARY_INV, 
							int blockSize = 11,
							double C = 15)
{
	if (grayImage.empty()) return grayImage;
	//blockSize must be an odd number,like 3,5,7...
	if (blockSize % 2 == 0) blockSize += 1;
	if (grayImage.channels() == 3) cv::cvtColor(grayImage, grayImage, cv::COLOR_BGR2GRAY);

	cv::adaptiveThreshold(grayImage, grayImage, maxval,adaptiveMethod, 
							thresholdType, blockSize,C);
	return grayImage;
}
cv::Mat otsu(cv::Mat grayImage, double maxval = 255)
{
	/*
	enum ThresholdTypes {
	THRESH_BINARY     = 0, //!< \f[\texttt{dst} (x,y) =  \fork{\texttt{maxval}}{if \(\texttt{src}(x,y) > \texttt{thresh}\)}{0}{otherwise}\f]
	THRESH_BINARY_INV = 1, //!< \f[\texttt{dst} (x,y) =  \fork{0}{if \(\texttt{src}(x,y) > \texttt{thresh}\)}{\texttt{maxval}}{otherwise}\f]
	THRESH_TRUNC      = 2, //!< \f[\texttt{dst} (x,y) =  \fork{\texttt{threshold}}{if \(\texttt{src}(x,y) > \texttt{thresh}\)}{\texttt{src}(x,y)}{otherwise}\f]
	THRESH_TOZERO     = 3, //!< \f[\texttt{dst} (x,y) =  \fork{\texttt{src}(x,y)}{if \(\texttt{src}(x,y) > \texttt{thresh}\)}{0}{otherwise}\f]
	THRESH_TOZERO_INV = 4, //!< \f[\texttt{dst} (x,y) =  \fork{0}{if \(\texttt{src}(x,y) > \texttt{thresh}\)}{\texttt{src}(x,y)}{otherwise}\f]
	THRESH_MASK       = 7,
	THRESH_OTSU       = 8, //!< flag, use Otsu algorithm to choose the optimal threshold value
	THRESH_TRIANGLE   = 16 //!< flag, use Triangle algorithm to choose the optimal threshold value

	*/

	if (grayImage.empty()) return grayImage;
	if (grayImage.channels() == 3) cv::cvtColor(grayImage, grayImage, cv::COLOR_BGR2GRAY);
	cv::threshold(grayImage, grayImage, 0, 255, 8);

	return grayImage;
}

cv::Mat open(cv::Mat binaryImage, int ksize=3, int kernelMode = 0)
{
	/*	MORPH_ERODE
		MORPH_DILATE
		MORPH_OPEN		dst = open(src, element) = dilate(erode(src, element))
		MORPH_CLOSE		dst = close(src, element) = erode(dilate(src, element))
		MORPH_GRADIENT		dst = morph_grad(src, element) = dilate(src, element)−erode(src, element)
		MORPH_TOPHAT		dst = tophat(src, element) = src−open(src, element)
		MORPH_BLACKHAT		dst = blackhat(src, element) = close(src, element)−src
		MORPH_HITMISS
		"hit or miss" . - Only supported for CV_8UC1 binary images.
		A tutorial canbe found in the documentation
	*/

	/*
	enum MorphTypes{
    MORPH_ERODE    = 0, 
    MORPH_DILATE   = 1, 
    MORPH_OPEN     = 2,                 
    MORPH_CLOSE    = 3, 
    MORPH_GRADIENT = 4,     
    MORPH_TOPHAT   = 5,       
    MORPH_BLACKHAT = 6,     
    MORPH_HITMISS  = 7  
     };
	*/
	/*
	enum MorphShapes {
    MORPH_RECT    = 0, //!< a rectangular structuring element:  \f[E_{ij}=1\f]
    MORPH_CROSS   = 1, //!< a cross-shaped structuring element:
                       //!< \f[E_{ij} =  \fork{1}{if i=\texttt{anchor.y} or j=\texttt{anchor.x}}{0}{otherwise}\f]
    MORPH_ELLIPSE = 2 //!< an elliptic structuring element, that is, a filled ellipse inscribed
                      //!< into the rectangle Rect(0, 0, esize.width, 0.esize.height)
	};
	*/
	if (binaryImage.empty() || binaryImage.channels() == 3) return binaryImage;
	cv::Mat element = cv::getStructuringElement(kernelMode, cv::Size(ksize, ksize));
	cv::morphologyEx(binaryImage, binaryImage, 2, element);
	return binaryImage;
}

cv::Mat close(cv::Mat binaryImage, int ksize = 3, int kernelMode = 2)
{
	/*	MORPH_ERODE
		MORPH_DILATE
		MORPH_OPEN		dst = open(src, element) = dilate(erode(src, element))
		MORPH_CLOSE		dst = close(src, element) = erode(dilate(src, element))
		MORPH_GRADIENT		dst = morph_grad(src, element) = dilate(src, element)−erode(src, element)
		MORPH_TOPHAT		dst = tophat(src, element) = src−open(src, element)
		MORPH_BLACKHAT		dst = blackhat(src, element) = close(src, element)−src
		MORPH_HITMISS
		"hit or miss" . - Only supported for CV_8UC1 binary images.
		A tutorial canbe found in the documentation
	*/

	/*
	enum MorphTypes{
	MORPH_ERODE    = 0,
	MORPH_DILATE   = 1,
	MORPH_OPEN     = 2,
	MORPH_CLOSE    = 3,
	MORPH_GRADIENT = 4,
	MORPH_TOPHAT   = 5,
	MORPH_BLACKHAT = 6,
	MORPH_HITMISS  = 7
	 };
	*/
	/*
	enum MorphShapes {
	MORPH_RECT    = 0, //!< a rectangular structuring element:  \f[E_{ij}=1\f]
	MORPH_CROSS   = 1, //!< a cross-shaped structuring element:
					   //!< \f[E_{ij} =  \fork{1}{if i=\texttt{anchor.y} or j=\texttt{anchor.x}}{0}{otherwise}\f]
	MORPH_ELLIPSE = 2 //!< an elliptic structuring element, that is, a filled ellipse inscribed
					  //!< into the rectangle Rect(0, 0, esize.width, 0.esize.height)
	};
	*/
	if (binaryImage.empty() || binaryImage.channels() == 3) return binaryImage;
	cv::Mat element = cv::getStructuringElement(kernelMode, cv::Size(ksize, ksize));
	cv::morphologyEx(binaryImage, binaryImage, 3, element);
	return binaryImage;
}

cv::Mat erode(cv::Mat binaryImage, int ksize = 3, int kernelMode = 0)
{
	/*	MORPH_ERODE
		MORPH_DILATE
		MORPH_OPEN		dst = open(src, element) = dilate(erode(src, element))
		MORPH_CLOSE		dst = close(src, element) = erode(dilate(src, element))
		MORPH_GRADIENT		dst = morph_grad(src, element) = dilate(src, element)−erode(src, element)
		MORPH_TOPHAT		dst = tophat(src, element) = src−open(src, element)
		MORPH_BLACKHAT		dst = blackhat(src, element) = close(src, element)−src
		MORPH_HITMISS
		"hit or miss" . - Only supported for CV_8UC1 binary images.
		A tutorial canbe found in the documentation
	*/

	/*
	enum MorphTypes{
	MORPH_ERODE    = 0,
	MORPH_DILATE   = 1,
	MORPH_OPEN     = 2,
	MORPH_CLOSE    = 3,
	MORPH_GRADIENT = 4,
	MORPH_TOPHAT   = 5,
	MORPH_BLACKHAT = 6,
	MORPH_HITMISS  = 7
	 };
	*/
	/*
	enum MorphShapes {
	MORPH_RECT    = 0, //!< a rectangular structuring element:  \f[E_{ij}=1\f]
	MORPH_CROSS   = 1, //!< a cross-shaped structuring element:
					   //!< \f[E_{ij} =  \fork{1}{if i=\texttt{anchor.y} or j=\texttt{anchor.x}}{0}{otherwise}\f]
	MORPH_ELLIPSE = 2 //!< an elliptic structuring element, that is, a filled ellipse inscribed
					  //!< into the rectangle Rect(0, 0, esize.width, 0.esize.height)
	};
	*/
	if (binaryImage.empty() || binaryImage.channels() == 3) return binaryImage;
	cv::Mat element = cv::getStructuringElement(kernelMode, cv::Size(ksize, ksize));
	cv::morphologyEx(binaryImage, binaryImage, 0
, element);
	return binaryImage;
}

cv::Mat dilate(cv::Mat binaryImage, int ksize = 3, int kernelMode = 0)
{
	/*	MORPH_ERODE
		MORPH_DILATE
		MORPH_OPEN		dst = open(src, element) = dilate(erode(src, element))
		MORPH_CLOSE		dst = close(src, element) = erode(dilate(src, element))
		MORPH_GRADIENT		dst = morph_grad(src, element) = dilate(src, element)−erode(src, element)
		MORPH_TOPHAT		dst = tophat(src, element) = src−open(src, element)
		MORPH_BLACKHAT		dst = blackhat(src, element) = close(src, element)−src
		MORPH_HITMISS
		"hit or miss" . - Only supported for CV_8UC1 binary images.
		A tutorial canbe found in the documentation
	*/

	/*
	enum MorphTypes{
	MORPH_ERODE    = 0,
	MORPH_DILATE   = 1,
	MORPH_OPEN     = 2,
	MORPH_CLOSE    = 3,
	MORPH_GRADIENT = 4,
	MORPH_TOPHAT   = 5,
	MORPH_BLACKHAT = 6,
	MORPH_HITMISS  = 7
	 };
	*/
	/*
	enum MorphShapes {
	MORPH_RECT    = 0, //!< a rectangular structuring element:  \f[E_{ij}=1\f]
	MORPH_CROSS   = 1, //!< a cross-shaped structuring element:
					   //!< \f[E_{ij} =  \fork{1}{if i=\texttt{anchor.y} or j=\texttt{anchor.x}}{0}{otherwise}\f]
	MORPH_ELLIPSE = 2 //!< an elliptic structuring element, that is, a filled ellipse inscribed
					  //!< into the rectangle Rect(0, 0, esize.width, 0.esize.height)
	};
	*/
	if (binaryImage.empty() || binaryImage.channels() == 3) return binaryImage;
	cv::Mat element = cv::getStructuringElement(kernelMode, cv::Size(ksize, ksize));
	cv::morphologyEx(binaryImage, binaryImage, 1, element);
	return binaryImage;
}

std::vector<std::vector<cv::Point>> findContours(cv::Mat binaryImage, 
	int topologyMode = 1, int contoursType = 1)
{
	/** Contour retrieval modes */
	/*enum
	{
		CV_RETR_EXTERNAL = 0,
		CV_RETR_LIST = 1,
		CV_RETR_CCOMP = 2,
		CV_RETR_TREE = 3,
		CV_RETR_FLOODFILL = 4
	};*/

	/** Contour approximation methods */
	/*enum
	{
		CV_CHAIN_CODE = 0,
		CV_CHAIN_APPROX_NONE = 1,
		CV_CHAIN_APPROX_SIMPLE = 2,
		CV_CHAIN_APPROX_TC89_L1 = 3,
		CV_CHAIN_APPROX_TC89_KCOS = 4,
		CV_LINK_RUNS = 5
	};*/
	
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	if (binaryImage.empty() || binaryImage.channels() == 3) return contours;
	cv::findContours(binaryImage, contours, topologyMode, contoursType);

	return contours;

}

cv::Mat drawContours(cv::Mat img, std::vector<std::vector<cv::Point>> contours,
	int contoursIdx = -1, int b=0, int g=0, int r=255)
{
	if (img.empty()) return img;
	cv::drawContours(img, contours, contoursIdx, cv::Scalar(b,g,r));
	return img;
}

cv::Mat watersherd(cv::Mat colorImage)
{
	if (colorImage.empty()) return colorImage;
	cv::Mat markersImage = cv::Mat::zeros(colorImage.rows, 
											colorImage.cols,
											CV_8UC1);
	cv::Mat grayImage;
	cv::cvtColor(colorImage, grayImage, cv::COLOR_BGR2GRAY);
	grayImage = adaptiveThreshold(grayImage);

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(grayImage, contours, 1, 2);

	for (int i = 0, j=0; i < contours.size(); i++,j++)
	{
		if (j > 255) j = 1;
		cv::drawContours(markersImage, contours, i, cv::Scalar::all(j+1));
	}
	cv::watershed(colorImage, markersImage);
	return markersImage;
}

cv::Mat bgr2gray(cv::Mat img)
{
	if (img.empty() || img.channels()!=3) return img;
	cv::Mat gray;
	cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
	return gray;
}

int accessPixels(cv::Mat img)
{
	for (int i = 0; i < img.rows; i++)
	{
		const uchar *ptr = img.ptr<uchar>(i);// i is the ith row's address,
		for (int j = 0; j < img.cols; j++)
		{
			if (img.channels() == 3)
			{
				int b=0, g=0, r=0;
				b = *(ptr++);
				g = *(ptr++);
				r = *(ptr++);
				//std::cout << "[b,g,r]=" << b << "," << g << "," << r << "\n";

			}
			if (img.channels() == 1)
			{
				int gray = 0;
				gray = *(ptr++);
				//std::cout <<gray<< "\n";
			}
		}
	}

	return 0;
}
cv::Mat bilateralFilter(cv::Mat srcImage, int d=1, 
						           double sigmaColor=1, 
						double sigmaSpace=1)
{
	/*void cv::bilateralFilter(InputArray 	src,
		OutputArray 	dst,
		int 	d,
		double 	sigmaColor,
		double 	sigmaSpace,
		int 	borderType = BORDER_DEFAULT
	)*/
	/*Parameters
		src	Source 8 - bit or floating - point, 1 - channel or 3 - channel image.
		dst	Destination image of the same size and type as src .
		d	Diameter of each pixel neighborhood that is used during filtering.If it is non - positive, it is computed from sigmaSpace.
		sigmaColor	Filter sigma in the color space.A larger value of the parameter means that farther colors within the pixel neighborhood(see sigmaSpace) will be mixed together, resulting in larger areas of semi - equal color.
		sigmaSpace	Filter sigma in the coordinate space.A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough(see sigmaColor).When d > 0, it specifies the neighborhood size regardless of sigmaSpace.Otherwise, d is proportional to sigmaSpace.
		borderType	border mode used to extrapolate pixels outside of the image, see BorderTypes
		*/
	cv::bilateralFilter(srcImage, srcImage, d, sigmaColor, sigmaSpace);
	return srcImage;
}
