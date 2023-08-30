
#include <cstdio>
#include <chrono>

#include "Dataset.h"
#include "Network.h"

int main() {
	
	DataSet trainingSet("../datasets/train-images-idx3-ubyte", "../datasets/train-labels-idx1-ubyte");
	Network net(trainingSet.getImgWidth() * trainingSet.getImgHeight(), 10, 16, 2);

	const int BATCH_SIZE = 500;
	const int BATCH_NUM = trainingSet.getSize() / BATCH_SIZE;
	int batch = 0;
	double accuracy = 0;

	std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
	while (accuracy < 0.99) {
		printf("Computing batch %i ... ", batch);
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
		double elapsedTime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - startTime).count();
		printf("Finished in %f\nCost: %f Accuracy: %f%\n\n", elapsedTime, avgCost, accuracy * 100);
		batch++;
		if (batch == BATCH_NUM) {
			batch = 0;
		}
	}

	char save = 0;
	std::cout << "Would you like to save this model? (y) ";
	std::cin >> save;
	std::cin.clear();

	if (save == 'y') {
		std::string filename;
		std::cin >> filename;
		std::cin.clear();
		net.saveModel(filename);
	}


}
