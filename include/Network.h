
#include <vector>

#include <Eigen/Dense>

class Network {
public:

	struct Layer {
	// public:

		Layer(int dim, int lastLayerDim);

	// private:

		int dim;

		Eigen::VectorXd activations;
		Eigen::VectorXd biases;
		Eigen::MatrixXd weights; // from the previous layer to this

	};

  Network(int inputLayerDim, int outputLayerDim, int hiddenLayerDim, int hiddenLayers);

	void initRandom();
	Eigen::VectorXd evaluate(const Eigen::VectorXd& input);
	void backPropagate(Eigen::VectorXd& gradient, int gi, int l, Eigen::VectorXd pdA);
	void offsetControls(const Eigen::VectorXd& offset);

	int getControlsSize();
	int getNumLayers() {return _layers.size();}

private:

	std::vector<Layer> _layers;

};