#include "opencv2\opencv.hpp"
#include "SurfObject.h"
#include "FeatureExtractor.h"
#include "SurfIdentifier.h"

using namespace SceneIdentifier;

int main(int argc, char** argv)
{
	//background and scene images
	cv::Mat bgImage = cv::imread("cardsBG.png", CV_LOAD_IMAGE_COLOR);
	cv::Mat sceneImage = cv::imread("cards.png", CV_LOAD_IMAGE_COLOR);

	//create the training card images and corresponding id sets
	std::vector<std::string> trainingCardImageURLs;
	trainingCardImageURLs.push_back("card7C.png");
	trainingCardImageURLs.push_back("card7D.png");
	trainingCardImageURLs.push_back("card8C.png");
	trainingCardImageURLs.push_back("card8D.png");
	trainingCardImageURLs.push_back("card6H.png");
	trainingCardImageURLs.push_back("card6S.png");
	trainingCardImageURLs.push_back("card7CInverted.png");
	trainingCardImageURLs.push_back("card7DInverted.png");
	trainingCardImageURLs.push_back("card8CInverted.png");
	trainingCardImageURLs.push_back("card8DInverted.png");
	trainingCardImageURLs.push_back("card6HInverted.png");
	trainingCardImageURLs.push_back("card6SInverted.png");

	std::vector<std::string> trainingCardIDs;
	trainingCardIDs.push_back("7C");
	trainingCardIDs.push_back("7D");
	trainingCardIDs.push_back("8C");
	trainingCardIDs.push_back("8D");
	trainingCardIDs.push_back("6H");
	trainingCardIDs.push_back("6S");
	trainingCardIDs.push_back("7C");
	trainingCardIDs.push_back("7D");
	trainingCardIDs.push_back("8C");
	trainingCardIDs.push_back("8D");
	trainingCardIDs.push_back("6H");
	trainingCardIDs.push_back("6S");

	//initialise the scene identifier with the initial image of the scene to use for background subtraction
	SurfIdentifier identifier(3, 4, 0.9);

	//create the training suft objects used for identifying different objects within a scene
	identifier.createTrainingObjects(trainingCardImageURLs, trainingCardIDs);

	//update the scene identifier with the background image
	identifier.update(bgImage);

	//update the scene identifier with the current scene
	identifier.update(sceneImage);

	//render and display the results
	cv::Mat renderImage(sceneImage);
	identifier.drawSceneMatchRects(renderImage, CV_RGB(0, 255, 0), 4);
	identifier.drawSceneMatchLabels(renderImage, CV_RGB(0, 0, 0), CV_RGB(255, 255, 255));
	cv::imshow("card detection", renderImage);
	cv::waitKey(0);
	
	return 0;
}