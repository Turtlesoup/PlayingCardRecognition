#ifndef SCENE_IDENTIFIER_H
#define SCENE_IDENTIFIER_H

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "FeatureExtractor.h"
#include "SurfObject.h"
#include <vector>
#include <map>

namespace SceneIdentifier
{
	class SurfIdentifier
	{
	public:
		SurfIdentifier(int history, int nmixtures, double backgroundRatio, double noiseSigma = 0.0);
		virtual ~SurfIdentifier();
		void createTrainingObjects(const std::vector<std::string> &imageURLs, const std::vector<std::string> &ids);
		void update(const cv::Mat &sceneImage);
		std::vector<SurfObject> &getTrainingObjects();
		std::vector<std::string> &getSceneObjectsIDs();
		std::map<std::string, SurfObject> &getSceneObjectIDToSceneObjectDictionary();
		std::map<std::string, cv::Point> &getSceneObjectIDToRegionPositionsDictionary();
		std::map<std::string, std::string> &getSceneObjectIDToBestMatchIDDictionary();
		std::map<std::string, unsigned int> &getSceneObjectIDToBestMatchIndexDictionary();
		std::map<std::string, std::vector<std::vector<cv::DMatch>>> &getSceneObjectIDToBestMatchesDictionary();
		cv::Mat estimateHomography(const std::string &sceneObjectID);
		void drawSceneMatchLabels(cv::Mat &renderImage, const cv::Scalar &labelColour, const cv::Scalar &textColour, int fontface = cv::FONT_HERSHEY_SIMPLEX, double scale = 0.5, int thickness = 1, int baseline = 0);
		void drawSceneMatchRects(cv::Mat &renderImage, const cv::Scalar &lineColour, float thickness);
	private:
		std::vector<SurfObject> trainingObjects;
		FeatureExtractor featureExtractor;
		std::vector<std::string> sceneObjectsIDs;
		std::map<std::string, SurfObject> sceneObjectIDToSceneObject;
		std::map<std::string, cv::Point> sceneObjectIDToRegionPositions;
		std::map<std::string, std::string> sceneObjectIDToBestMatchID;
		std::map<std::string, unsigned int> sceneObjectIDToBestMatchIndex;
		std::map<std::string, std::vector<std::vector<cv::DMatch>>> sceneObjectIDToBestMatches;
	};
}

#endif