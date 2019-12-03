#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <iterator>

using namespace std;
using namespace cv;


string noImagesFolder = "Images/no/";
string yesImagesFolder = "Images/yes/";
string meanHistogramsFileName = "Mean Histograms.txt";


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

vector<vector<long>> CalculateImagesHistograms(vector<Mat> images)
{
	vector<vector<long>> histograms;

	for (int imageNumber = 0; imageNumber < images.size(); imageNumber++)
	{
		vector<long> histogram(256);
		
		int histSize = 256;
		float range[] = { 0, 255 };
		const float* ranges[] = { range };

		MatND hist;
		auto image = images[imageNumber];
		calcHist(&image, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false);

		for (int h = 0; h < histSize; h++)
		{
			histogram[h] = hist.at<float>(h);
		}

		histograms.push_back(histogram);
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

vector<vector<int>> createEmptyHistograms(int blocksCount)
{
	vector<vector<int>> meanHistograms;
	for (int i = 0; i < blocksCount; i++)
	{
		vector<int> hist;
		for (int j = 0; j < 256; j++)
			hist.push_back(0);
		meanHistograms.push_back(hist);
	}
	return meanHistograms;
}

int main()
{
	int rowsCount = 8;
	int colsCount = 8;
	int blocksCount = rowsCount * colsCount;
	int imagesCount = 10;

	bool predictImage = true;
	int predictDiffSize = 1000;

	vector<vector<int>> meanHistograms = createEmptyHistograms(blocksCount);
	if (!predictImage)
	{
		for (int i = 0; i < imagesCount; i++)
		{
			Mat sourceImage = imread(noImagesFolder + "/" + to_string(i) + ".jpg", 0);
			auto imageBlocksWithNo = SplitImageToBlocks(sourceImage, rowsCount, colsCount);
			auto histogramsWithNo = CalculateImagesHistograms(imageBlocksWithNo);

			for (int k = 0; k < blocksCount; k++)
			{
				for (int j = 0; j < 256; j++)
				{
					meanHistograms[k][j] += (histogramsWithNo[k][j] / imagesCount);
				}
			}
		}


		ofstream oHistFile;
		oHistFile.open(meanHistogramsFileName);
		if (oHistFile.is_open())
		{
			oHistFile.clear();
			for (int k = 0; k < blocksCount; k++)
			{
				string line;
				for (int i = 0; i < 256; i++)
				{
					line += (to_string(meanHistograms[k][i]) + " ");
				}
				oHistFile << line;
				oHistFile << "\n";
			}
			oHistFile.close();
		}
	}
	else
	{
		ifstream iHistFile;
		iHistFile.open(meanHistogramsFileName);
		if (iHistFile.is_open())
		{
			int histNumber = 0;
			string line;
			while (getline(iHistFile, line))
			{
				vector<int> result;
				istringstream iss(line);
				for (string s; iss >> s; )
					result.push_back(atoi(s.c_str()));
				meanHistograms[histNumber++] = result;
			}
		}


		for (int i = 0; i < imagesCount; i++)
		{
			Mat sourceImage = imread(yesImagesFolder + "/" + to_string(i) + ".jpg", 0);
			auto imageBlocksWithNo = SplitImageToBlocks(sourceImage, rowsCount, colsCount);
			auto histogramsWithNo = CalculateImagesHistograms(imageBlocksWithNo);

			for (int k = 0; k < blocksCount; k++)
			{
				bool withProblem = false;
				for (int j = 0; j < 256; j++)
				{
					int diff = abs(meanHistograms[k][j] - histogramsWithNo[k][j]);
					if (diff > predictDiffSize) {
						withProblem = true;
						break;
					}
				}
				if (withProblem)
				{
					cout << "Image #" << i << " has a problem" << endl;
					break;
				}
			}
			cout << endl;
		}
	}

	waitKey(0);
}

