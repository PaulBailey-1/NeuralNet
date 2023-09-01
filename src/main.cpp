
#include <cstdio>
#include <chrono>
#include <vector>

#include <matplot/matplot.h>

#include "Dataset.h"
#include "Network.h"

int main(int argc, char* argv[]) {
	
	DataSet trainingSet("../datasets/train-images-idx3-ubyte", "../datasets/train-labels-idx1-ubyte");
	Network net(trainingSet.getImgWidth() * trainingSet.getImgHeight(), 10, 16, 2);

	const int BATCH_SIZE = 500;
	const int BATCH_NUM = trainingSet.getSize() / BATCH_SIZE;
	int batch = 0;
	double accuracy = 0.0;
	std::vector<double> accuracies;
	std::vector<double> times = {0.0};
	double lastPlotTime = 0.0;
	double avgBatchTime = 0.0;
	int lastLogBatch = 0;
	int deltaBatches = 0;
	std::chrono::time_point<std::chrono::high_resolution_clock> startTime;

	matplot::xlabel("Time (s)");
	matplot::ylabel("Accuracy (%)");

	printf("Training...\n");
	while (accuracy < 0.9) {
		startTime = std::chrono::high_resolution_clock::now();
		Eigen::VectorXd avgGradient = Eigen::VectorXd::Zero(net.getControlsSize());
		double avgCost = 0;
		for (int img = 0; img < BATCH_SIZE; img++) {

			int idx = batch * BATCH_SIZE + img;
			Eigen::VectorXd output = net.evaluate(trainingSet.getImg(idx));

			int result, truth;
			output.maxCoeff(&result);
			trainingSet.getLabel(idx).maxCoeff(&truth);
			accuracy += truth == result ? 1 : 0;

			Eigen::VectorXd pdA = output - trainingSet.getLabel(idx); // A - y
			avgCost += pdA.array().square().sum();
			net.backPropagate(avgGradient, 0, net.getNumLayers() - 1, pdA);
		}
		avgGradient /= BATCH_SIZE;
		avgCost /= BATCH_SIZE;
		accuracy /= BATCH_SIZE;
		net.offsetControls(-avgGradient);

		double elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startTime).count() / 1000.0;
		avgBatchTime = (avgBatchTime + elapsedTime) / 2.0;

		times.push_back(times.back() + elapsedTime);
		accuracies.push_back(accuracy * 100);
		deltaBatches++;
		if (times.back() - lastPlotTime > 5.0) {
			matplot::plot(times, accuracies);
			lastPlotTime = times.back();
			printf("Finished %i batches (%i to %i)\nAvg. Time: %f Cost: %f Accuracy: %f%\n\n", deltaBatches, lastLogBatch, batch, avgBatchTime, avgCost, accuracy * 100);
			lastLogBatch = batch;
			deltaBatches = 0;
		}

		batch++;
		if (batch == BATCH_NUM) {
			batch = 0;
		}
	}
	
	printf("Testing...\n");
	DataSet testingSet("../datasets/t10k-images-idx3-ubyte", "../datasets/t10k-labels-idx1-ubyte");

	startTime = std::chrono::high_resolution_clock::now();
	accuracy = 0.0;
	for (int i = 0; i < testingSet.getSize(); i++) {
		int result, truth;
		net.evaluate(testingSet.getImg(i)).maxCoeff(&result);
		testingSet.getLabel(i).maxCoeff(&truth);
		accuracy += result == truth ? 1 : 0;
	}
	accuracy /= testingSet.getSize();
	double elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startTime).count() / 1000.0;
	printf("Tested %i cases in %fs\nAccuracy: %f%\n\n", testingSet.getSize(), elapsedTime, accuracy * 100);

	net.saveModel("../models/model");

}
