#include <iostream>
#include <opencv2/opencv.hpp>


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

vector<Mat> CalculateImagesHistograms(vector<Mat> images, vector<Mat> &outHists)
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

		outHists.push_back(histogram);

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

vector<Mat> diff(vector<Mat> histsWithYes, vector<Mat> histsWithNo) {
	vector<Mat> result;
	for (int i = 0; i < histsWithYes.size(); i++) {
		Mat diff;
		absdiff(histsWithNo[i], histsWithYes[i], diff);
		result.push_back(diff);
	}
	return result;
}

int main()
{
	int rowsCount = 2;
	int colsCount = 2;
	Mat sourceImage = imread("no/1 no.jpeg");
	bool showImageBlocks = false;
	bool showBlocksHistograms = false;


	Mat greyImageWithNo;
	cvtColor(sourceImage, greyImageWithNo, COLOR_BGR2GRAY);

	vector<Mat> histsWithNo;
	vector<Mat> imageBlocksWithNo = SplitImageToBlocks(greyImageWithNo, rowsCount, colsCount);
	vector<Mat> histogramsWithNo = CalculateImagesHistograms(imageBlocksWithNo, histsWithNo);

	Mat withYes = imread("yes/Y1.jpg");
	Mat geyImageWithYes;
	cvtColor(withYes, geyImageWithYes, COLOR_BGR2GRAY);
	vector<Mat> histsWithYes;
	vector<Mat> imageBlocksWithYes = SplitImageToBlocks(geyImageWithYes, rowsCount, colsCount);
	vector<Mat> histogramsWithYes = CalculateImagesHistograms(imageBlocksWithYes, histsWithYes);

	vector<Mat> difference = diff(histsWithYes, histsWithNo);


	if (showImageBlocks)
	{
		imshow("Source Image", sourceImage);
		ShowAllImages(imageBlocksWithNo, "Image block");
	}
	if (showBlocksHistograms)
	{
		ShowAllImages(histogramsWithNo, "Histogram");
	}

	waitKey(0);
}

