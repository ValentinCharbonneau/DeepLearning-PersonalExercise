#if !defined(NN_H)
#define NN_H

typedef struct {
    int inputNodes;
    int hiddenNodes;
    int outputNodes;
    int trainingSet;

    double learningRate;

    double* hiddenLayer;
    double* outputLayer;
    double* hiddenLayerBias;
    double* outputLayerBias;

    double** hiddenWeights;
    double** outputWeights;
} NeuralNetwork;

NeuralNetwork* createNeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, int trainingSet, double learningRate);
void trainNeuralNetwork(NeuralNetwork* nn, double** inputs, double** targets);
void backPropagation(NeuralNetwork* nn, double* inputs, double* targets);
void finalPrediction(NeuralNetwork* nn, double* inputs, double* targets);
void forwardPropagation(NeuralNetwork* nn, double* inputs);
void freeNeuralNetwork(NeuralNetwork* nn);

#endif