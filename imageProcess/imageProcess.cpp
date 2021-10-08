#include "imageProcess.h"


bool cvbag::isImageEmpty(const cv::Mat &image, const std::string functionName)
{
	if (image.empty())
	{
		std::cout << "ERROR \t in cvbag::" << functionName << " ,the input image is empty!\n";
		return true;
	}

	return false;
}

int cvbag::getAllImagePath(std::string folder, std::vector<cv::String> &imagePathList, bool flg)
{
	cv::glob(folder, imagePathList, flg);
	return 0;
}

int cvbag::showImage(const cv::Mat &image, const std::string winName, const int waitKeyMode, const int destroyMode)
{
	if (cvbag::isImageEmpty(image, "showImage"))
	{
		return -1;
	}

	cv::namedWindow(winName, 0);
	cv::imshow(winName, image);
	cv::waitKey(waitKeyMode);
	if (destroyMode == 1)
	{
		cv::destroyWindow(winName);
	}

	return 0;
}

int cvbag::gaussBlur(const cv::Mat &image, cv::Mat &dst, const int ksize, double sigma)
{
	if (cvbag::isImageEmpty(image, "gaussBlur"))
	{
		return -1;
	}
	cv::GaussianBlur(image, dst, cv::Size(ksize, ksize), sigma);
	return 0;
}

int cvbag::sobel_x(const cv::Mat &image, cv::Mat &dst, const int ksize)
{
	if (cvbag::isImageEmpty(image, "sobel_x"))
	{
		return -1;
	}

	cv::Sobel(image, dst, CV_64F, 1, 0, ksize);
	return 0;
}

int cvbag::sobel_y(const cv::Mat &image, cv::Mat &dst, const int ksize)
{
	if (cvbag::isImageEmpty(image, "sobel_y"))
	{
		return -1;
	}

	cv::Sobel(image, dst, CV_64F, 0, 1, ksize);
	return 0;
}

int cvbag::sobel_xy(const cv::Mat &image, cv::Mat &dst, const int ksize)
{
	if (cvbag::isImageEmpty(image, "sobel_xy"))
	{
		return -1;
	}
	cv::Mat sobel_x, sobel_y;
	cvbag::sobel_x(image, sobel_x, ksize);
	cvbag::sobel_y(image, sobel_y, ksize);
	cv::addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0, dst);
	return 0;
}

int cvbag::canny(const cv::Mat &image, cv::Mat &dst, const int low, const int heigh)
{
	if (cvbag::isImageEmpty(image, "canny")) return -1;

	cv::Canny(image, dst, low, heigh);
	return 0;
}

int cvbag::otsu(const cv::Mat &image, cv::Mat &dst)
{
	if (cvbag::isImageEmpty(image, "adaptiveThreshold")) return -1;
	if (image.channels() == 3) cv::cvtColor(image, dst, cv::COLOR_BGR2GRAY);
	cv::threshold(dst, dst, 0, 255, 8);
	return 0;
}

int cvbag::threshold(const cv::Mat &image, cv::Mat &dst, const int th, const int mode, const int maxval)
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

	if (cvbag::isImageEmpty(image, "adaptiveThreshold")) return -1;
	if (image.channels() == 3)
	{
		cv::cvtColor(image, dst, cv::COLOR_BGR2GRAY);
		cv::threshold(dst, dst, th, maxval, mode);
	}
	else {
		cv::threshold(image, dst, th, maxval, mode);
	}

	return 0;
}

int cvbag::adaptiveThreshold(const cv::Mat &image, cv::Mat &dst, int blockSize, double C, double maxval,
	int adaptiveMethod, int thresholdType)
{
	if (cvbag::isImageEmpty(image, "adaptiveThreshold")) return -1;
	//blockSize must be an odd number,like 3,5,7...
	if (blockSize % 2 == 0) blockSize += 1;
	if (image.channels() == 3) {
		cv::cvtColor(image, dst, cv::COLOR_BGR2GRAY);
		cv::adaptiveThreshold(dst, dst, maxval, adaptiveMethod, thresholdType, blockSize, C);
	} 
	else {
		cv::adaptiveThreshold(image, dst, maxval, adaptiveMethod, thresholdType, blockSize, C);
	}

	return 0;
}

int  cvbag::findContours(const cv::Mat &binaryImage, std::vector<std::vector<cv::Point>> &contours, int topologyMode, int contoursType)
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

	if (cvbag::isImageEmpty(binaryImage, "findContours")) return -1;

	std::vector<cv::Vec4i> hierarchy;
	if (binaryImage.channels() == 3) return -1;
	cv::findContours(binaryImage, contours, topologyMode, contoursType);

	return 0;
}

int cvbag::drawContours(cv::Mat &image, const std::vector<std::vector<cv::Point>> &contours, int contoursIdx, int b, int g, int r)
{
	if (cvbag::isImageEmpty(image, "drawContours")) return -1;
	cv::drawContours(image, contours, contoursIdx, cv::Scalar(b, g, r));
	return 0;
}

int cvbag::dilate(const cv::Mat &binaryImage, cv::Mat &dst, const  int ksize, const int kernelMode)
{
	/*  MORPH_ERODE
		MORPH_DILATE
		MORPH_OPEN      dst = open(src, element) = dilate(erode(src, element))
		MORPH_CLOSE     dst = close(src, element) = erode(dilate(src, element))
		MORPH_GRADIENT      dst = morph_grad(src, element) = dilate(src, element)−erode(src, element)
		MORPH_TOPHAT        dst = tophat(src, element) = src−open(src, element)
		MORPH_BLACKHAT      dst = blackhat(src, element) = close(src, element)−src
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
	if (cvbag::isImageEmpty(binaryImage, "dilate")) return -1;
	if (binaryImage.channels() == 3) return -1;
	cv::Mat element = cv::getStructuringElement(kernelMode, cv::Size(ksize, ksize));
	cv::morphologyEx(binaryImage, dst, 1, element);
	return 0;
}

int cvbag::erode(const cv::Mat &binaryImage, cv::Mat &dst, const int ksize, const int kernelMode)
{
	/*  MORPH_ERODE
		MORPH_DILATE
		MORPH_OPEN      dst = open(src, element) = dilate(erode(src, element))
		MORPH_CLOSE     dst = close(src, element) = erode(dilate(src, element))
		MORPH_GRADIENT      dst = morph_grad(src, element) = dilate(src, element)−erode(src, element)
		MORPH_TOPHAT        dst = tophat(src, element) = src−open(src, element)
		MORPH_BLACKHAT      dst = blackhat(src, element) = close(src, element)−src
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
	if (cvbag::isImageEmpty(binaryImage, "erode")) return -1;
	if (binaryImage.channels() == 3) return -1;
	cv::Mat element = cv::getStructuringElement(kernelMode, cv::Size(ksize, ksize));
	cv::morphologyEx(binaryImage, dst, 0, element);
	return 0;
}

int cvbag::open(const cv::Mat &binaryImage, cv::Mat &dst, const int ksize, const int kernelMode)
{

	if (cvbag::isImageEmpty(binaryImage, "erode")) return -1;
	if (binaryImage.channels() == 3) return -1;
	cv::Mat element = cv::getStructuringElement(kernelMode, cv::Size(ksize, ksize));
	cv::morphologyEx(binaryImage, dst, 2, element);
	return 0;
}

int cvbag::close(const cv::Mat &binaryImage, cv::Mat &dst, const int ksize, const int kernelMode)
{

	if (cvbag::isImageEmpty(binaryImage, "erode")) return -1;
	if (binaryImage.channels() == 3) return -1;
	cv::Mat element = cv::getStructuringElement(kernelMode, cv::Size(ksize, ksize));
	cv::morphologyEx(binaryImage, dst, 3, element);
	return 0;
}

int cvbag::rotateImage(const cv::Mat &srcImage, cv::Mat &dst, const double angle, const int mode)
{
	//mode = 0 ,Keep the original image size unchanged
	//mode = 1, Change the original image size to fit the rotated scale, padding with zero
	if (cvbag::isImageEmpty(srcImage, "rotateImage")) return -1;
	if (mode == 0)
	{
		cv::Point2f center((srcImage.cols - 1) / 2.0, (srcImage.rows - 1) / 2.0);
		cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
		cv::warpAffine(srcImage, dst, rot, srcImage.size());//the original size
	}
	else {

		double alpha = -angle * CV_PI / 180.0;//convert angle to radian format 

		cv::Point2f srcP[3];
		cv::Point2f dstP[3];
		srcP[0] = cv::Point2f(0, srcImage.rows);
		srcP[1] = cv::Point2f(srcImage.cols, 0);
		srcP[2] = cv::Point2f(srcImage.cols, srcImage.rows);

		//rotate the pixels
		for (int i = 0; i < 3; i++)
			dstP[i] = cv::Point2f(srcP[i].x*cos(alpha) - srcP[i].y*sin(alpha), srcP[i].y*cos(alpha) + srcP[i].x*sin(alpha));
		double minx, miny, maxx, maxy;
		minx = std::min(std::min(std::min(dstP[0].x, dstP[1].x), dstP[2].x), float(0.0));
		miny = std::min(std::min(std::min(dstP[0].y, dstP[1].y), dstP[2].y), float(0.0));
		maxx = std::max(std::max(std::max(dstP[0].x, dstP[1].x), dstP[2].x), float(0.0));
		maxy = std::max(std::max(std::max(dstP[0].y, dstP[1].y), dstP[2].y), float(0.0));

		int w = maxx - minx;
		int h = maxy - miny;

		//translation
		for (int i = 0; i < 3; i++)
		{
			if (minx < 0)
				dstP[i].x -= minx;
			if (miny < 0)
				dstP[i].y -= miny;
		}

		cv::Mat warpMat = cv::getAffineTransform(srcP, dstP);
		cv::warpAffine(srcImage, dst, warpMat, cv::Size(w, h));//extend size

	}//end else

	return 0;
}

int cvbag::match::cpuTemplateMatch(const cv::Mat &srcImage, const cv::Mat &tempImage, cv::Mat &result,
	double &matchVal, cv::Point &matchLoc, int mode)
{
	if (srcImage.empty() || tempImage.empty())
	{
		std::cout << "ERROR:In function cpuTemplateMatch: input image is empty! \n";
		return -1;
	}

	//cv::Mat result;

	int result_w = srcImage.cols - tempImage.cols;
	int result_h = srcImage.rows - tempImage.rows;
	if (result_w < 0 || result_h < 0)
	{
		std::cout << "ERROR:in function opencvTemplateMatch: roi image's size should be larger than tamplate's \n";
		return -1;
	}
	//result.create(result_h, result_w, CV_32FC1);
	switch (mode)
	{
	case 0:
		//R = sum (t-Roi)^2
		cv::matchTemplate(srcImage, tempImage, result, cv::TM_SQDIFF);
		cv::minMaxLoc(result, &matchVal, NULL, &matchLoc, NULL);
		break;
	case 1:
		//R = sum (t-Roi)^2/(sqrt(sum t^2   *  sum Roi^2))
		cv::matchTemplate(srcImage, tempImage, result, cv::TM_SQDIFF_NORMED);
		cv::minMaxLoc(result, &matchVal, NULL, &matchLoc, NULL);
		break;
	case 2:
		//R = sum t*Roi
		cv::matchTemplate(srcImage, tempImage, result, cv::TM_CCORR);
		cv::minMaxLoc(result, NULL, &matchVal, NULL, &matchLoc);
		break;
	case 3:
		//R = sum t*Roi / (sqrt(sum t^2   *  sum Roi^2))
		cv::matchTemplate(srcImage, tempImage, result, cv::TM_CCORR_NORMED);
		cv::minMaxLoc(result, NULL, &matchVal, NULL, &matchLoc);
		break;
	case 4:
		//R = sum t1*Roi1
		//t1 = t - t_mean
		//Roi1 = Roi - Roi_mean
		cv::matchTemplate(srcImage, tempImage, result, cv::TM_CCOEFF);
		cv::minMaxLoc(result, NULL, &matchVal, NULL, &matchLoc);
		break;
	case 5:
		//R = sum t1*Roi1 / (sqrt(sum t1^2   *  sum Roi1^2))
		//t1 = t - t_mean
		//Roi1 = Roi - Roi_mean
		cv::matchTemplate(srcImage, tempImage, result, cv::TM_CCOEFF_NORMED);
		cv::minMaxLoc(result, NULL, &matchVal, NULL, &matchLoc);
		break;
	default:
		cv::matchTemplate(srcImage, tempImage, result, cv::TM_CCOEFF_NORMED);
		cv::minMaxLoc(result, NULL, &matchVal, NULL, &matchLoc);
		break;
	}

	return 0;
}


int cpuTemplateMatchWithAngle(const cv::Mat &srcImage, const cv::Mat &tempImage, cv::Mat &result,
	double &matchVal, cv::Point &matchLoc, int mode,
	double angleStart , double angleEnd , double angleStep )
{
	return 0;
}

int cvbag::gammaTransform_threeChannels(const cv::Mat &image, cv::Mat &dst, const int table[])
{
	std::vector<cv::Mat> channelsImage;
	std::vector<cv::Mat> channelsImage_dst;
	cv::split(image, channelsImage);
	for (int i = 0; i < channelsImage.size(); i++)
	{
		channelsImage_dst.push_back(cv::Mat::zeros(cv::Size(image.cols, image.rows), CV_8UC1));

	}

	for (int i = 0; i < channelsImage.size(); i++)
	{
		gammaTransform(channelsImage[i], channelsImage_dst[i], table);
	}
	cv::merge(channelsImage_dst, dst);

	return 0;
}

int cvbag::getGammaTable(const double gamma, int *a, const int num)
{
	for (int i = 0; i < num; i++)
	{
		a[i] = int(pow(i / 255.0, gamma) *255.0);
	}

	return 0;
}

int cvbag::gamma(const cv::Mat &image, cv::Mat &dst, const double gamma)
{
	if (image.empty()) return -1;

	int table[256] = { 0 };
	getGammaTable(gamma, table);

	gammaTransform(image, dst, table);


	return 0;
}

int cvbag::gammaTransform(const cv::Mat &image, cv::Mat &dst, const int  table[])
{
	if (image.empty()) return -1;

	if (image.channels() == 3)
	{
		gammaTransform_threeChannels(image, dst, table);
	}

	if (dst.empty()) dst = cv::Mat::zeros(cv::Size(image.cols, image.rows), CV_8UC1);

	const uchar * ps = NULL;
	uchar * pd = NULL;

	for (int i = 0; i < image.rows; i++)
	{
		ps = image.ptr<uchar>(i);
		pd = dst.ptr<uchar>(i);
		for (int j = 0; j < image.cols; j++)
		{
			*(pd + j) = table[int(*(ps + j))];
		}
	}
	return 0;
}

int cvbag::gamma_picewiseLinear(const cv::Mat &image, cv::Mat &dst, const int src1, const int dst1, const int src2, const int dst2)
{
	if (image.empty()) return -1;

	int table[256] = { 0 };
	if (src1 <= 0 | src1 > 255 | src1 >= src2 | src2 > 255 | dst1 < 0 | dst1 > 255 | dst2 < 0 | dst2 > 255) return -1;

	getGammaTable_piecewiseLinear(table, src1, dst1, src2, dst2);

	gammaTransform(image, dst, table);

	return 0;
}

int cvbag::getGammaTable_piecewiseLinear(int *a, const int src1, const int dst1, const int src2, const int dst2)
{
	for (int i = 0; i < src1; i++)
	{
		a[i] = int(float(dst1) / float(src1)*i);
	}

	for (int i = src1; i < src2; i++)
	{
		a[i] = int(float(dst2 - dst1) / float(src2 - src1) * (i - src1) + dst1);
	}

	for (int i = src2; i < 256; i++)
	{
		a[i] = int(float(255 - dst2) / float(255 - src2) * (i - src2) + dst2);
	}

	return 0;
}






