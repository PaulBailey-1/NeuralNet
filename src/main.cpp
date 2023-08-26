
#include <cstdio>

#include "Dataset.h"
#include "Network.h"

int main() {
	
	DataSet trainingSet("../datasets/train-images-idx3-ubyte", "../datasets/train-labels-idx1-ubyte");
	Network net(trainingSet.getImgWidth() * trainingSet.getImgHeight(), 10, 16, 2);

	const int BATCH_SIZE = 500;
	const int BATCH_NUM = trainingSet.getSize() / BATCH_SIZE;
	int batch = 0;
	while (1) { // while cost is high
		Eigen::VectorXd avgGradient(net.getControlsSize()).zeros();
		double avgCost = 0;
		for (int img = 0; img < BATCH_SIZE; img++) {
			int idx = batch * BATCH_SIZE + img;
			Eigen::VectorXd pdA = net.evaluate(trainingSet.getImg(idx)) - trainingSet.getLabel(idx); // A - y
			avgCost += pdA.array().square().sum();
			net.backPropagate(avgGradient, 0, net.getNumLayers() - 1, pdA);
		}
		avgGradient /= BATCH_SIZE;
		avgCost /= BATCH_SIZE;
		// log cost
		// Apply avg gradient to network, subtracting
		batch++;
		if (batch == BATCH_NUM) {
			batch = 0;
		}
	}

}
