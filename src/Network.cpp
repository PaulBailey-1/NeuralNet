
#include <cstdlib>
#include <ctime>

#include "Network.h"

// get random double between -1 and 1
double getRand() {
    return (((double) std::rand()) - RAND_MAX / 2.0) / (RAND_MAX / 2.0);
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

Eigen::VectorXd sigmoid(Eigen::VectorXd x) {
    Eigen::VectorXd out(x.rows());
    for (int i = 0; i < x.rows(); i++) {
        out(i) = sigmoid(x(i));
    }
    return out;
}

double sigmoidDeriv(double x) {
    double ex = exp(x);
    return ex / ((ex + 1) * (ex + 1));
}

Network::Network(int inputLayerDim, int outputLayerDim, int hiddenLayerDim, int hiddenLayers) {
    _layers.push_back(Layer(inputLayerDim, 0));
    int previousLayerDim = inputLayerDim;
    for (int i = 0; i < hiddenLayers; i++) {
        _layers.push_back(Layer(hiddenLayerDim, previousLayerDim));
        previousLayerDim = hiddenLayerDim;
    }
    _layers.push_back(Layer(outputLayerDim, previousLayerDim));
}

// deprecated
void Network::initRandom() {
    std::srand(std::time(nullptr));
    for (int i = 0; i < _layers.size(); i++) {
        for (int j = 0; _layers[i].dim; j++) {
            _layers[i].activations(j) = getRand();
            _layers[i].biases(j) = getRand();
            for (int k = 0; _layers[i].weights.cols(); k++) {
                _layers[i].weights(i, k) = getRand();
            }
        }
    }
}

Eigen::VectorXd Network::evaluate(const Eigen::VectorXd& input) {
    if (input.size() != _layers.front().activations.size()) {
        printf("Error: input dimension mismatch");
    }
    _layers.front().activations = input;
    for (int i = 1; i < _layers.size(); i++) {
        _layers[i].activations = sigmoid(_layers[i].weights * _layers[i-1].activations + _layers[i].biases);
    }
    return _layers.back().activations;
}

// Recursivly back propagates through layers, appending partial derivs of cost for weights & biases to the gradient vector
// l = layer index, gi = gradient index, pdA = A - y, y = desired activations, j = layer(l) neuron index, k = layer(l-1) neuron index
void Network::backPropagate(Eigen::VectorXd& gradient, int gi, int l, Eigen::VectorXd pdA) {

    // Find desired changes to weights & biases, pd of w across L and L-1 & vec of pd ob b across L appended to gradient
    Eigen::VectorXd z = _layers[l].weights * _layers[l-1].activations + _layers[l].biases; // same as above
    for (int j = 0; j < _layers[l].dim; j++) {
        double pdBj = 2 * pdA(j) * sigmoidDeriv(z(j)); // p deriv of C/Bj
        gradient(gi) += pdBj;
        for (int k = 0; k < _layers[l-1].dim; k++) {
            gradient(gi) += pdBj * _layers[l-1].activations(k); // p deriv of C/Wjk
        }
    }

    if (l == 1) {
        return;
    }
    // Find the desired activations changes of l-1
    Eigen::VectorXd next_pdA = Eigen::VectorXd::Zero(_layers[l-1].dim); // p deriv of C/A across k
    for (int k = 0; k < _layers[l-1].dim; k++) {
        for (int j = 0; j < _layers[l].dim; j++) {
            next_pdA(k) += 2 * pdA(j) * sigmoidDeriv(z(j)) * _layers[l].weights(j, k); // repeated above
        }
    }
    backPropagate(gradient, gi + _layers[l].dim * (_layers[l-1].dim + 1), l - 1, next_pdA);

}

void Network::offsetControls(const Eigen::VectorXd& offset) {
    int i = 0;
    for (int l = _layers.size() - 1; l > 0; l--) {
        Eigen::MatrixXd combined;
        for (int j = 0; j < _layers[l].dim; j++) {
            Eigen::VectorXd seg = offset.segment(i, _layers[l-1].dim);
            i += _layers[l-1].dim;
            combined.row(j) = seg.transpose();
        }
        _layers[l].biases += combined.col(0);
        _layers[l].weights += combined.rightCols(combined.rows() - 1);
    }
}

int Network::getControlsSize() {
    static int size = 0;
    if (size == 0) {
        for (Layer& layer : _layers) {
            size += layer.activations.size() + layer.weights.size();
        }
    }
    return size;
}

Network::Layer::Layer(int dim, int lastLayerDim) {
    dim = dim;
    activations = Eigen::VectorXd::Random(dim);
    biases = Eigen::VectorXd::Random(dim);
    weights = Eigen::MatrixXd::Random(dim, lastLayerDim);
}