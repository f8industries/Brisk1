/**
* @file MatchTemplate_Demo.cpp
* @brief Sample code to use the function MatchTemplate
* @author OpenCV team
*/

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

#include "opencv2\features2d\features2d.hpp"

using namespace std;
using namespace cv;

const char* image_window = "Source Image";
const char* map_window = "Map window";

/// Global Variables
Mat img; Mat map; Mat result;
Mat grayImg; Mat grayMap;

int main(int, char** argv)
{
	cout << "1" << endl;

	img = imread("C://test//input_trees.JPG", 1);
	map = imread("C://test//input_stitch.JPG", 1);

	if (img.data == NULL || map.data == NULL)
	{
		cout << "NO IMAGEges" << endl;
		waitKey(0);
		return -3;
	};

	cvtColor(img, grayImg, CV_BGR2GRAY);
	cvtColor(map, grayMap, CV_BGR2GRAY);

	///// Create windows
	namedWindow(image_window, WINDOW_AUTOSIZE);
	//namedWindow(map_window, WINDOW_AUTOSIZE);

	cout << "GOT IMAGE" << endl;

	//imshow(image_window, grayImg);
	//imshow(map_window, map);

	///////////////
	std::vector<cv::KeyPoint> keypointsA, keypointsB;
	cv::Mat descriptorsA, descriptorsB;
	//////////////////
	int Threshl = 40;
	int Octaves = 4;// (pyramid layer) from which the keypoint has been extracted
	float PatternScales = 1.0f;
	///////////////////

	Ptr<BRISK> brisk = BRISK::create(Threshl, Octaves, PatternScales);
	brisk->detect(grayImg, keypointsA);
	brisk->compute(grayImg, keypointsA, descriptorsA);

	brisk->detect(grayMap, keypointsB);
	brisk->compute(grayMap, keypointsB, descriptorsB);

	//BFMatcher matcher(cv::Hamming);
	//cv::BFMatcher matcher(cv::NORM_L2, false);
	//Ptr<cv::DescriptorMatcher> matcher = BFMatcher::create(cv::NORM_L2)
	BFMatcher matcher(NORM_L2, false);

	vector<DMatch> matches;

	matcher.match(descriptorsA, descriptorsB, matches);

	cout << "matches no " << matches.size();

	cv::Mat all_matches, all_matches_small;
	cv::drawMatches(grayImg, keypointsA, grayMap, keypointsB,
		matches, all_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
		vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	resize(all_matches, all_matches_small, Size(), 0.4, 0.4, INTER_LINEAR);

	cv::imshow("BRISK All Matches", all_matches_small);

	imwrite("C://test//out.JPG", all_matches_small);

	waitKey(0);
	return 0;
}

