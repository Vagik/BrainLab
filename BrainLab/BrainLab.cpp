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

vector<vector<float>> CalculateImagesHistograms(vector<Mat> images)
{
	vector<vector<float>> histograms;

	for (int imageNumber = 0; imageNumber < images.size(); imageNumber++)
	{
		vector<float> histogram(256);
		
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

vector<vector<float>> createEmptyHistograms(int blocksCount)
{
	vector<vector<float>> meanHistograms;
	for (int i = 0; i < blocksCount; i++)
	{
		vector<float> hist;
		for (int j = 0; j < 256; j++)
			hist.push_back(0);
		meanHistograms.push_back(hist);
	}
	return meanHistograms;
}

void writeHistogramsToFile(vector<vector<float>> meanHistograms, int blocksCount)
{
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

void trainOrPredictMethod(bool predictImage, int imagesCount, int rowsCount, int colsCount, int predictDiffSize)
{
	int blocksCount = rowsCount * colsCount;
	vector<vector<float>> meanHistograms = createEmptyHistograms(blocksCount);

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

		writeHistogramsToFile(meanHistograms, blocksCount);
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
				vector<float> result;
				istringstream iss(line);
				for (string s; iss >> s; )
					result.push_back(stof(s.c_str()));
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
}

int main()
{
	const int imagesCount = 30;
	const bool predictImage = true;
	
	int startPredictDiffSize = 100;
	int finishPredictDiffSize = 1500;
	int predictDiffStep = 50;

	int startBlocksCountInRow = 2;
	int finishBlocksCountInRow = 20;

	vector<double> results;

	ofstream oHistFile;
	oHistFile.open(meanHistogramsFileName);
	if (oHistFile.is_open())
	{
		oHistFile.clear();
		for (int countInRow = startBlocksCountInRow; countInRow < finishBlocksCountInRow; countInRow++)
		{
			oHistFile << "BLOCKS NUMBER IN A ROW = " << countInRow << "\n";

			int rowsCount = countInRow;
			int colsCount = countInRow;
			int blocksCount = rowsCount * colsCount;

			vector<vector<float>> meanHistograms = createEmptyHistograms(blocksCount);

			// Train
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


			// Predict
			for (int predictDiffSize = startPredictDiffSize; predictDiffSize < finishPredictDiffSize; predictDiffSize += predictDiffStep)
			{
				int correctYesPredictions = 0;
				for (int i = 0; i < imagesCount; i++)
				{
					Mat sourceImage = imread(yesImagesFolder + "/" + to_string(i) + ".jpg", 0);
					auto imageBlocksWithYes = SplitImageToBlocks(sourceImage, rowsCount, colsCount);
					auto histogramsWithYes = CalculateImagesHistograms(imageBlocksWithYes);

					for (int k = 0; k < blocksCount; k++)
					{
						bool withProblem = false;
						for (int j = 0; j < 256; j++)
						{
							int diff = abs(meanHistograms[k][j] - histogramsWithYes[k][j]);
							if (diff > predictDiffSize)
							{
								correctYesPredictions++;
								withProblem = true;
								break;
							}
						}
						if (withProblem)
						{
							break;
						}
					}
				}

				int incorrectNoPredictions = 0;
				for (int i = 0; i < imagesCount; i++)
				{
					Mat sourceImage = imread(noImagesFolder + "/" + to_string(i) + ".jpg", 0);
					auto imageBlocksWithNo = SplitImageToBlocks(sourceImage, rowsCount, colsCount);
					auto histogramsWithNo = CalculateImagesHistograms(imageBlocksWithNo);

					for (int k = 0; k < blocksCount; k++)
					{
						bool withProblem = false;
						for (int j = 0; j < 256; j++)
						{
							int diff = abs(meanHistograms[k][j] - histogramsWithNo[k][j]);
							if (diff > predictDiffSize)
							{
								incorrectNoPredictions++;
								withProblem = true;
								break;
							}
						}
						if (withProblem)
						{
							break;
						}
					}
				}
				int correctNoPredictions = imagesCount - incorrectNoPredictions;

				double result = ((double)(correctYesPredictions + correctNoPredictions) / (double)(2 * imagesCount));
				results.push_back(result);
				oHistFile << "Prediction diff value = " << predictDiffSize << "; Prediction result = " << result << "\n";
			}
			oHistFile << "---------------------------------------------------------------------------\n\n";
		}

		double max = 0;
		for (int i = 0; i < results.size(); i++)
		{
			if (results[i] > max) {
				max = results[i];
			}
		}
		oHistFile << "THE BEST PREDICTION RESULT: " << max << "\n";
	}
	oHistFile.close();
}

