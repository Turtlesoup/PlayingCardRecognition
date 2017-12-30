#ifndef SurfObject_H
#define SurfObject_H

#include "opencv2/core/core.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/highgui/highgui.hpp"

namespace SceneIdentifier
{
	class SurfObject
	{
	public:
		SurfObject();
		SurfObject(cv::Mat imageMat, std::string objectID);
		virtual ~SurfObject();
		std::string getObjectID();
		cv::Mat getImage();
		cv::Mat getDescriptors();
		std::vector<cv::KeyPoint> getKeypoints();
		unsigned int getClosestMatch(std::vector<SurfObject> &candidates, std::vector<std::vector<cv::DMatch>> &SurfObjectGoodMatches);
		float getSurfPointMatches(cv::Mat &trainSurfPoints, cv::Mat &testSurfPoints, std::vector<cv::DMatch> &goodMatches, float thresholdMultiplier);

		SurfObject& SurfObject::operator=(SurfObject arg)
		{
			id = arg.getObjectID();
			image = arg.getImage();
			descriptors = arg.getDescriptors();
			keyPoints = arg.getKeypoints();
			return *this;
		}

	private:
		void getSurfPoints();
		std::string id;
		cv::Mat image;
		cv::Mat descriptors;
		std::vector<cv::KeyPoint> keyPoints;
		const int MIN_HESSIAN;
	};
}

#endif