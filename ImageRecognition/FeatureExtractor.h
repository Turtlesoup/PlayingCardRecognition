#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2\opencv.hpp"

namespace SceneIdentifier
{
	class FeatureExtractor
	{
	public:
		FeatureExtractor(int history, int nmixtures, double backgroundRatio, double noiseSigma = 0.0);
		virtual ~FeatureExtractor();
		void addFrame(cv::Mat frameImage);
		void getRegionOfInterestsFromScene(const cv::Mat &sceneImage, std::vector<cv::Mat> &sceneRegionsOfInterest, std::vector<cv::Point> &sceneCardRegionPositions);
	private:
		cv::Ptr<cv::BackgroundSubtractor> MOG;
		cv::Mat MODMask;
	};
}

#endif