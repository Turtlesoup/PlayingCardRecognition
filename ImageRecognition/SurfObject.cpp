#include "SurfObject.h"

namespace SceneIdentifier
{
	SurfObject::SurfObject() : MIN_HESSIAN(400)
	{

	}

	SurfObject::SurfObject(cv::Mat imageMat, std::string objectID) : MIN_HESSIAN(400),
		id(objectID),
		image(imageMat)
	{
		getSurfPoints();
	}

	SurfObject::~SurfObject()
	{
	}

	std::string SurfObject::getObjectID()
	{
		return id;
	}

	cv::Mat SurfObject::getImage()
	{
		return image;
	}

	cv::Mat SurfObject::getDescriptors()
	{
		return descriptors;
	}

	std::vector<cv::KeyPoint> SurfObject::getKeypoints()
	{
		return keyPoints;
	}

	void SurfObject::getSurfPoints()
	{
		cv::SurfFeatureDetector detector(MIN_HESSIAN);
		cv::SurfDescriptorExtractor extractor;
		detector.detect(image, keyPoints);
		extractor.compute(image, keyPoints, descriptors);
	}

	unsigned int SurfObject::getClosestMatch(std::vector<SurfObject> &candidates, std::vector<std::vector<cv::DMatch>> &SurfObjectGoodMatches)
	{
		unsigned int bestMatchIndex;
		float lowestMatchesAverageDistances = 1;

		unsigned int candidatesIndex;
		unsigned int candidatesLength = candidates.size();
		for (candidatesIndex = 0; candidatesIndex < candidatesLength; ++candidatesIndex)
		{
			std::vector<cv::DMatch> goodMatches;

			float averageDistance = getSurfPointMatches(candidates[candidatesIndex].getDescriptors(), getDescriptors(), goodMatches, 10);
			SurfObjectGoodMatches.push_back(goodMatches);

			if (averageDistance < lowestMatchesAverageDistances)
			{
				lowestMatchesAverageDistances = averageDistance;
				bestMatchIndex = candidatesIndex;
			}
		}

		return bestMatchIndex;
	}

	float SurfObject::getSurfPointMatches(cv::Mat &trainSurfPoints, cv::Mat &testSurfPoints, std::vector<cv::DMatch> &goodMatches, float thresholdMultiplier)
	{
		//find the good matches between the object and the image surf points
		cv::FlannBasedMatcher matcher;
		std::vector<cv::DMatch> matches;
		matcher.match(trainSurfPoints, testSurfPoints, matches);

		//compute the min and max values for the matches
		double maxDist = 0; double minDist = 100;
		int surfPointIndex = 0;
		for (surfPointIndex = 0; surfPointIndex < trainSurfPoints.rows; ++surfPointIndex)
		{
			double dist = matches[surfPointIndex].distance;
			if (dist < minDist)
			{
				minDist = dist;
			}
			if (dist > maxDist)
			{
				maxDist = dist;
			}
		}

		//get the good matches and sum the distances of all the matches
		float matchPointDistanceSum = 0;
		for (int surfPointIndex = 0; surfPointIndex < trainSurfPoints.rows; ++surfPointIndex)
		{
			if (matches[surfPointIndex].distance <= thresholdMultiplier * minDist)
			{
				goodMatches.push_back(matches[surfPointIndex]);
			}

			matchPointDistanceSum += matches[surfPointIndex].distance;
		}

		//return the average of the match distances
		matchPointDistanceSum /= (float)matches.size();
		return matchPointDistanceSum;
	}
}