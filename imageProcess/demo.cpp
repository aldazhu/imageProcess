#include "imageProcess.h"


int templateMatchDemo()
{
	const std::string fileName = "K:\\imageData\\colorR\\test2.bmp";
	cv::Mat srcImage = cv::imread(fileName);

	//cut a region from srcImage as a templateImage
	int x = 5, y = 5, w = 20, h = 20;
	cv::Mat templateImage = srcImage(cv::Rect(x, y, w, h));

	//match
	double matchVal;//score
	cv::Point matchLoc;//top left 
	cv::Mat result;//
	int mode = 1;
	cvbag::match::cpuTemplateMatch(srcImage, templateImage, result, matchVal, matchLoc, mode);

	//show result
	std::cout << "matchVal = " << matchVal << std::endl;
	cv::Point topLeft = matchLoc;
	cv::Point bottomRight = cv::Point(topLeft.x + templateImage.cols, topLeft.y + templateImage.rows);
	cv::Mat drawImage = cv::imread(fileName);
	cv::rectangle(drawImage, cv::Rect(topLeft, bottomRight), cv::Scalar(0, 255, 0), 2);

	cvbag::showImage(srcImage, "srcImage", 1);
	cvbag::showImage(templateImage, "tempImage", 1);
	cvbag::showImage(drawImage, "drawImage", 1);
	cv::normalize(result, result, 0, 1, cv::NORM_MINMAX);//result data type is 64F,transform to 0~1
	cvbag::showImage(result, "result", 0);
	return 0;
}

int demo()
{
	const std::string fileName = "K:\\imageData\\colorR\\test2.bmp";
	//const std::string fileName = "K:\\imageData\\lena\\Lena.png";

	//cvbag tool;

	//showImage
	cv::Mat image = cv::imread(fileName);
	cvbag::showImage(image, "image", 0, 0);

	//gauss smooth
	cv::Mat gauss;
	cvbag::gaussBlur(image, gauss);
	cvbag::showImage(gauss, "gauss");

	//sobel edge detect
	cv::Mat sobel_xy;
	cvbag::sobel_y(gauss, sobel_xy);
	////convert format CV_64F to CV_8U ,so that the image can use cv::imshow to show image.
	cv::convertScaleAbs(sobel_xy, sobel_xy);
	cvbag::showImage(sobel_xy, "sobel_xy");

	//otsu
	cv::Mat otsu;
	cvbag::otsu(gauss, otsu);
	cvbag::showImage(otsu, "otsu");

	//threshold
	cv::Mat threshold;
	cvbag::threshold(gauss, threshold, 120, 8 + 1);
	cvbag::showImage(threshold, "threshold");

	//adaptiveThreshold
	cv::Mat adaptiveThreshold;
	cvbag::adaptiveThreshold(gauss, adaptiveThreshold, 11, 8);
	cvbag::showImage(adaptiveThreshold, "adaptiveThreshold");

	//canny
	cv::Mat canny;
	cvbag::canny(image, canny, 120, 180);
	cvbag::showImage(canny, "canny");

	//contours
	std::vector<std::vector<cv::Point>> contours;
	cvbag::findContours(adaptiveThreshold, contours);
	cv::Mat conImage = gauss.clone();
	cvbag::drawContours(conImage, contours, -1, 0, 255, 0);
	cvbag::showImage(conImage, "contours");

	//dilate
	cv::Mat dilate;
	cvbag::dilate(otsu, dilate, 3, 0);
	cvbag::showImage(dilate, "dilate");

	//erode
	cv::Mat erode;
	cvbag::erode(otsu, erode, 3, 0);
	cvbag::showImage(erode, "erode");

	//open
	cv::Mat open;
	cvbag::open(otsu, open);
	cvbag::showImage(open, "open");

	//close
	cv::Mat close;
	cvbag::close(otsu, close);
	cvbag::showImage(close, "close");

	//rotateImage
	cv::Mat rotatedImage;
	cvbag::rotateImage(image, rotatedImage, 25, 1);
	cvbag::showImage(rotatedImage, "rotatedImage");

	//gamma
	cv::Mat gamma;
	cvbag::gamma(image, gamma, 2);
	cvbag::showImage(gamma, "gamma");

	//gamma_piecewiseLinaer
	cv::Mat gamma_piece;
	cvbag::gamma_picewiseLinear(image, gamma_piece, 120, 60, 160, 200);
	cvbag::showImage(gamma_piece, "gamma_piece");

	return 0;
}


int main()
{
	//demo();
	templateMatchDemo();

	return 0;
}