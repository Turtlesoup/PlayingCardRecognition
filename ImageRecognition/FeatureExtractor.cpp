#include "FeatureExtractor.h"

namespace SceneIdentifier
{
	FeatureExtractor::FeatureExtractor(int history, int nmixtures, double backgroundRatio, double noiseSigma) : MOG(new cv::BackgroundSubtractorMOG(history, nmixtures, backgroundRatio, noiseSigma))
	{
	}

	FeatureExtractor::~FeatureExtractor()
	{
	}

	void FeatureExtractor::addFrame(cv::Mat frameImage)
	{
		MOG->operator()(frameImage, MODMask);
	}

	void FeatureExtractor::getRegionOfInterestsFromScene(const cv::Mat &sceneImage, std::vector<cv::Mat> &sceneRegionsOfInterest, std::vector<cv::Point> &sceneCardRegionPositions)
	{
		cv::Size sceneImageSize = sceneImage.size();

		//create the background subtracted mask
		addFrame(sceneImage);

		//find all contours in the image
		CvSeq* contours;
		CvSeq* contourPoints;
		CvMemStorage *storage = cvCreateMemStorage(0);
		cvFindContours(&IplImage(MODMask), storage, &contours, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));

		//while there are contours
		while (contours)
		{
			//get the result pixels for the contours connected to each seperate object
			contourPoints = cvApproxPoly(contours, sizeof(CvContour), storage, CV_POLY_APPROX_DP, cvContourPerimeter(contours)*0.02, 0);

			//get the corner for the polygon
			CvPoint* cornerPoint;
			unsigned int minX = sceneImageSize.width;
			unsigned int maxX = 0;
			unsigned int minY = sceneImageSize.height;
			unsigned int maxY = 0;
			int totalPoints = contourPoints->total;
			for (int pointIndex = 0; pointIndex < totalPoints; ++pointIndex)
			{
				cornerPoint = (CvPoint*)cvGetSeqElem(contourPoints, pointIndex);

				if (cornerPoint->x < minX)
				{
					minX = cornerPoint->x;
				}
				else if (cornerPoint->x > maxX)
				{
					maxX = cornerPoint->x;
				}

				if (cornerPoint->y < minY)
				{
					minY = cornerPoint->y;
				}
				else if (cornerPoint->y > maxY)
				{
					maxY = cornerPoint->y;
				}
			}

			//get the bounding rect of the polygon on the image
			cv::Rect bounds(minX, minY, maxX - minX, maxY - minY);

			//create the sub image containing the found shape and add to
			//the list of regions within the scene to check for cards within.
			cv::Mat sceneRegionOfInterest = sceneImage(bounds);
			sceneRegionsOfInterest.push_back(sceneRegionOfInterest);
			sceneCardRegionPositions.push_back(cv::Point(minX, minY));

			//get the next set of contours
			contours = contours->h_next;
		}
	}
}