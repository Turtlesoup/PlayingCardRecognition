#include "SurfIdentifier.h"

namespace SceneIdentifier
{
	SurfIdentifier::SurfIdentifier(int history, int nmixtures, double backgroundRatio, double noiseSigma) : featureExtractor(history, nmixtures, backgroundRatio, noiseSigma)
	{
	}

	SurfIdentifier::~SurfIdentifier()
	{
	}

	void SurfIdentifier::createTrainingObjects(const std::vector<std::string> &imageURLs, const std::vector<std::string> &ids)
	{
		assert(imageURLs.size() == ids.size());

		unsigned int trainingObjectsIndex;
		unsigned int trainingObjectsLength = imageURLs.size();
		for (trainingObjectsIndex = 0; trainingObjectsIndex < trainingObjectsLength; ++trainingObjectsIndex)
		{
			trainingObjects.push_back(SurfObject(cv::imread(imageURLs[trainingObjectsIndex], CV_LOAD_IMAGE_COLOR), ids[trainingObjectsIndex]));
		}
	}

	void SurfIdentifier::update(const cv::Mat &sceneImage)
	{
		//clear the scene data vectors
		sceneObjectsIDs.clear();
		sceneObjectIDToSceneObject.clear();
		sceneObjectIDToRegionPositions.clear();
		sceneObjectIDToBestMatchID.clear();
		sceneObjectIDToBestMatchIndex.clear();
		sceneObjectIDToBestMatches.clear();

		//extract the SurfObject ROIs from the scene
		std::vector<SurfObject> sceneObjects;
		std::vector<cv::Point> sceneObjectRegionPositions;
		std::vector<cv::Mat> sceneSegmentImages;
		featureExtractor.getRegionOfInterestsFromScene(sceneImage, sceneSegmentImages, sceneObjectRegionPositions);

		//create the scene SurfObjects from the extracted SurfObject images
		char objectID[33];
		unsigned int sceneSurfObjectImagesIndex;
		unsigned int sceneSurfObjectImagesLength = sceneSegmentImages.size();
		for (sceneSurfObjectImagesIndex = 0; sceneSurfObjectImagesIndex < sceneSurfObjectImagesLength; ++sceneSurfObjectImagesIndex)
		{
			_itoa(sceneSurfObjectImagesIndex, objectID, 10);

			sceneObjects.push_back(SurfObject(sceneSegmentImages[sceneSurfObjectImagesIndex], objectID));
			sceneObjectsIDs.push_back(objectID);
		}

		//get the surf point matches for each detected SurfObject within tehs cene against each training SurfObject
		//and determine the most likely SurfObject that the scene SurfObject is based on the match results
		unsigned int sceneSurfObjectIndex;
		unsigned int sceneSurfObjectLength = sceneObjects.size();
		std::string sceneObjectID;
		for (sceneSurfObjectIndex = 0; sceneSurfObjectIndex < sceneSurfObjectLength; ++sceneSurfObjectIndex)
		{
			sceneObjectID = sceneObjects[sceneSurfObjectIndex].getObjectID();

			std::vector<std::vector<cv::DMatch>> objectsGoodMatches;
			unsigned int bestMatchIndex = sceneObjects[sceneSurfObjectIndex].getClosestMatch(trainingObjects, objectsGoodMatches);

			sceneObjectIDToSceneObject[sceneObjectID] = sceneObjects[sceneSurfObjectIndex];
			sceneObjectIDToRegionPositions[sceneObjectID] = sceneObjectRegionPositions[sceneSurfObjectIndex];
			sceneObjectIDToBestMatchID[sceneObjectID] = trainingObjects[bestMatchIndex].getObjectID();
			sceneObjectIDToBestMatchIndex[sceneObjectID] = bestMatchIndex;
			sceneObjectIDToBestMatches[sceneObjectID] = objectsGoodMatches;
		}
	}

	std::vector<SurfObject> &SurfIdentifier::getTrainingObjects()
	{
		return trainingObjects;
	}

	std::vector<std::string> &SurfIdentifier::getSceneObjectsIDs()
	{
		return sceneObjectsIDs;
	}

	std::map<std::string, SurfObject> &SurfIdentifier::getSceneObjectIDToSceneObjectDictionary()
	{
		return sceneObjectIDToSceneObject;
	}

	std::map<std::string, cv::Point> &SurfIdentifier::getSceneObjectIDToRegionPositionsDictionary()
	{
		return sceneObjectIDToRegionPositions;
	}

	std::map<std::string, std::string> &SurfIdentifier::getSceneObjectIDToBestMatchIDDictionary()
	{
		return sceneObjectIDToBestMatchID;
	}

	std::map<std::string, unsigned int> &SurfIdentifier::getSceneObjectIDToBestMatchIndexDictionary()
	{
		return sceneObjectIDToBestMatchIndex;
	}

	std::map<std::string, std::vector<std::vector<cv::DMatch>>> &SurfIdentifier::getSceneObjectIDToBestMatchesDictionary()
	{
		return sceneObjectIDToBestMatches;
	}

	cv::Mat SurfIdentifier::estimateHomography(const std::string &sceneObjectID)
	{
		unsigned int bestMatchIndex = sceneObjectIDToBestMatchIndex[sceneObjectID];

		std::vector<cv::Point2f> trainingMatchPoints;
		std::vector<cv::Point2f> sceneMatchPoints;

		//get the matche points for the training and scene objects
		std::vector<cv::DMatch> goodMatches = sceneObjectIDToBestMatches[sceneObjectID][bestMatchIndex];
		for (int matchIndex = 0; matchIndex < goodMatches.size(); ++matchIndex)
		{
			//get the keypoints from the good matches
			trainingMatchPoints.push_back(trainingObjects[bestMatchIndex].getKeypoints()[goodMatches[matchIndex].queryIdx].pt);
			sceneMatchPoints.push_back(sceneObjectIDToSceneObject[sceneObjectID].getKeypoints()[goodMatches[matchIndex].trainIdx].pt);
		}
		return findHomography(cv::Mat(trainingMatchPoints), cv::Mat(sceneMatchPoints), CV_RANSAC);
	}

	void SurfIdentifier::drawSceneMatchLabels(cv::Mat &renderImage, const cv::Scalar &labelColour, const cv::Scalar &textColour, int fontface, double scale, int thickness, int baseline)
	{
		unsigned int sceneCardIndex;
		unsigned int sceneCardLength = sceneObjectsIDs.size();
		for (sceneCardIndex = 0; sceneCardIndex < sceneCardLength; ++sceneCardIndex)
		{
			std::string sceneObjectID = sceneObjectsIDs[sceneCardIndex];
			unsigned int bestMatchIndex = sceneObjectIDToBestMatchIndex[sceneObjectID];

			cv::Point rectPosition = sceneObjectIDToRegionPositions[sceneObjectID];
			cv::Size rectSize = sceneObjectIDToSceneObject[sceneObjectID].getImage().size();
			cv::Rect rect = cv::Rect(rectPosition.x, rectPosition.y, rectSize.width, rectSize.height);
			std::string label = trainingObjects[bestMatchIndex].getObjectID();

			cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
			cv::Point textPosition(rect.x + (rect.width - text.width) / 2, rect.y + (rect.height + text.height) / 2);

			cv::rectangle(renderImage, textPosition + cv::Point(0, baseline), textPosition + cv::Point(text.width, -text.height), labelColour, CV_FILLED);
			putText(renderImage, label, textPosition, fontface, scale, textColour, thickness, 8);
		}
	}

	void SurfIdentifier::drawSceneMatchRects(cv::Mat &renderImage, const cv::Scalar &lineColour, float thickness)
	{
		unsigned int sceneCardIndex;
		unsigned int sceneCardLength = sceneObjectsIDs.size();
		for (sceneCardIndex = 0; sceneCardIndex < sceneCardLength; ++sceneCardIndex)
		{
			std::string sceneObjectID = sceneObjectsIDs[sceneCardIndex];
			unsigned int bestMatchIndex = sceneObjectIDToBestMatchIndex[sceneObjectID];
			cv::Point rectPosition = sceneObjectIDToRegionPositions[sceneObjectID];

			//draw bounding rectangle
			cv::Mat homography = estimateHomography(sceneObjectID);
			cv::Mat trainingObjectImage = trainingObjects[bestMatchIndex].getImage();

			std::vector<cv::Point2f> trainingObjectImageCorners(4);
			std::vector<cv::Point2f> sceneObjectImageCorners(4);

			//create training object corner points which are the corners of the training object image
			trainingObjectImageCorners[0] = cvPoint(0, 0);
			trainingObjectImageCorners[1] = cvPoint(trainingObjectImage.cols, 0);
			trainingObjectImageCorners[2] = cvPoint(trainingObjectImage.cols, trainingObjectImage.rows);
			trainingObjectImageCorners[3] = cvPoint(0, trainingObjectImage.rows);

			//transform training object corner points by the homography to obtain the scene object corner points
			perspectiveTransform(cv::Mat(trainingObjectImageCorners), cv::Mat(sceneObjectImageCorners), homography);

			line(renderImage, sceneObjectImageCorners[0] + cv::Point2f(rectPosition), sceneObjectImageCorners[1] + cv::Point2f(rectPosition), lineColour, thickness);
			line(renderImage, sceneObjectImageCorners[1] + cv::Point2f(rectPosition), sceneObjectImageCorners[2] + cv::Point2f(rectPosition), lineColour, thickness);
			line(renderImage, sceneObjectImageCorners[2] + cv::Point2f(rectPosition), sceneObjectImageCorners[3] + cv::Point2f(rectPosition), lineColour, thickness);
			line(renderImage, sceneObjectImageCorners[3] + cv::Point2f(rectPosition), sceneObjectImageCorners[0] + cv::Point2f(rectPosition), lineColour, thickness);
		}
	}
}