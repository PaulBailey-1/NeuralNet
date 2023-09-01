
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>

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

Network::Layer::Layer(int thisDim, int lastLayerDim) {
    dim = thisDim;
    activations = Eigen::VectorXd::Random(thisDim);
    biases = Eigen::VectorXd::Random(thisDim);
    weights = Eigen::MatrixXd::Random(thisDim, lastLayerDim);
}

Network::Network(int inputLayerDim, int outputLayerDim, int hiddenLayerDim, int hiddenLayers) {
    printf("Creating network... ");
    _layers.push_back(Layer(inputLayerDim, 0));
    int previousLayerDim = inputLayerDim;
    for (int i = 0; i < hiddenLayers; i++) {
        _layers.push_back(Layer(hiddenLayerDim, previousLayerDim));
        previousLayerDim = hiddenLayerDim;
    }
    _layers.push_back(Layer(outputLayerDim, previousLayerDim));
    printf("done\n");
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
        _layers[i].zActivations = _layers[i].weights * _layers[i-1].activations + _layers[i].biases;
        _layers[i].activations = sigmoid(_layers[i].zActivations);
    }
    return _layers.back().activations;
}

// Recursivly back propagates through layers, appending partial derivs of cost for weights & biases to the gradient vector
// l = layer index, gi = gradient index, pdA = A - y, y = desired activations, j = layer(l) neuron index, k = layer(l-1) neuron index
void Network::backPropagate(Eigen::VectorXd& gradient, int gi, int l, Eigen::VectorXd pdA) {

    // Find desired changes to weights & biases, pd of w across L and L-1 & vec of pd ob b across L appended to gradient
    Eigen::VectorXd pdB = Eigen::VectorXd(_layers[l].dim);
    for (int j = 0; j < _layers[l].dim; j++) {
        double pdBj = 2 * pdA(j) * sigmoidDeriv(_layers[l].zActivations(j)); // p deriv of C/Bj
        gradient(gi) += pdBj;
        gi++;
        for (int k = 0; k < _layers[l-1].dim; k++) {
            gradient(gi) += pdBj * _layers[l-1].activations(k); // p deriv of C/Wjk
            gi++;
        }
    }

    if (l == 1) {
        return;
    }
    // Find the desired activations changes of l-1
    Eigen::VectorXd next_pdA = Eigen::VectorXd::Zero(_layers[l-1].dim); // p deriv of C/A across k
    for (int k = 0; k < _layers[l-1].dim; k++) {
        for (int j = 0; j < _layers[l].dim; j++) {
            next_pdA(k) += 2 * pdA(j) * sigmoidDeriv(_layers[l].zActivations(j)) * _layers[l].weights(j, k); // repeated above
        }
    }
    backPropagate(gradient, gi, l - 1, next_pdA);

}

void Network::offsetControls(const Eigen::VectorXd& offset) {
    int i = 0;
    for (int l = _layers.size() - 1; l > 0; l--) {
        Eigen::MatrixXd combined(_layers[l].dim, _layers[l-1].dim + 1);
        for (int j = 0; j < _layers[l].dim; j++) {
            Eigen::VectorXd seg = offset.segment(i, _layers[l-1].dim + 1);
            i += _layers[l-1].dim + 1;
            combined.row(j) = seg.transpose();
        }
        _layers[l].biases += combined.col(0);
        _layers[l].weights += combined.rightCols(combined.cols() - 1);
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

void Network::saveModel(std::string filename)  {
    printf("Saving model to %s ... ", filename.c_str());
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile.is_open()) {
        printf("Error: Could not create file: '%s'", filename.c_str());
        return;
    }
    
    int numLayers = _layers.size();
    outFile.write((char*) &numLayers, sizeof(int));
    for (Layer& layer : _layers) {
        int layerDim = layer.dim;
        outFile.write((char*) &numLayers, sizeof(int));
        outFile.write((char*) layer.biases.data(), sizeof(double) * layer.dim);
        if (layer.weights.cols() != 0) {
            outFile.write((char*) layer.weights.data(), sizeof(double) * layer.weights.rows() * layer.weights.cols());
        }
    }
    outFile.close();
    printf("Saved\n");
}