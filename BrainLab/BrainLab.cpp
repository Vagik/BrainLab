#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;


void ShowAllImages(vector<Mat> images, string dialogTitle)
{
	ostringstream stringStream;
	stringStream << dialogTitle;
	stringStream << " %i";
	for (int i = 0; i < images.size(); i++)
	{
        imshow(format(stringStream.str().c_str(), i + 1), images[i]);
	}
}


vector<Mat> SplitImageToBlocks(Mat sourceImage, int rowsCount, int colsCount)
{
	int blocksCount = colsCount * rowsCount;
	vector<Mat> imageBlocks(blocksCount);
	int blockWidth = sourceImage.cols / colsCount;
	int blockHeight = sourceImage.rows / rowsCount;

	int blockNumber = 0;
	for (int row = 0; row < rowsCount; row++)
	{
		for (int col = 0; col < colsCount; col++)
		{
			int rectX = col * blockWidth;
			int rectY = row * blockHeight;
			int rectWidth = blockWidth;
			int rectHeight = blockHeight;

			if (col == colsCount - 1)
			{
				rectWidth = sourceImage.cols - (col)* blockWidth;
			}
			if (row == rowsCount - 1)
			{
				rectHeight = sourceImage.rows - (row)* blockHeight;
			}
			imageBlocks[blockNumber++] = sourceImage(Rect(rectX, rectY, rectWidth, rectHeight));
		}
	}
	return imageBlocks;
}

vector<Mat> CalculateImagesHistograms(vector<Mat> images)
{
	vector<Mat> histograms;

	for (int imageNumber = 0; imageNumber < images.size(); imageNumber++)
	{
		int histSize = 256;
		float range[] = { 0, 256 };
		const float* histRange = { range };
		int histWidth = 400;
		int histHeight = 400;
		int binWidth = cvRound((double)histWidth / histSize);

		Mat histogram;

		calcHist(&images[imageNumber], 1, 0, Mat(), histogram, 1, &histSize, &histRange, true, false);

		Mat histImage(histHeight, histWidth, CV_8UC3, Scalar(0, 0, 0));
		normalize(histogram, histogram, 0, histImage.rows, NORM_MINMAX, -1, Mat());

		for (int i = 1; i < histSize; i++)
		{
			line(histImage, Point(binWidth * (i - 1), histHeight - cvRound(histogram.at<float>(i - 1))),
				Point(binWidth * (i), histHeight - cvRound(histogram.at<float>(i))),
				Scalar(255, 0, 0), 1, 8, 0);
		}
		histograms.push_back(histImage);
	}
	return histograms;
}

int main()
{
	int rowsCount = 2;
	int colsCount = 2;
	Mat sourceImage = imread("Images/lena.jpg");
	bool showImageBlocks = true;
	bool showBlocksHistograms = true;


	Mat greyImage;
	cvtColor(sourceImage, greyImage, COLOR_BGR2GRAY);

	vector<Mat> imageBlocks = SplitImageToBlocks(greyImage, rowsCount, colsCount);
	vector<Mat> histograms = CalculateImagesHistograms(imageBlocks);
	
	imshow("Source Image", sourceImage);


	if (showImageBlocks)
	{
		ShowAllImages(imageBlocks, "Image block");
	}
	if (showBlocksHistograms)
	{
		ShowAllImages(histograms, "Histogram");
	}
	
	waitKey(0);
}