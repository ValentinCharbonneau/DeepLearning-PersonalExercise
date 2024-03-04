#include "headers/includes.h"
#include "headers/sigmoid.h"
#include "headers/weight.h"
#include "headers/nn.h"

NeuralNetwork* createNeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, int trainingSet, double learningRate) {
    NeuralNetwork* nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    nn->inputNodes = inputNodes;
    nn->hiddenNodes = hiddenNodes;
    nn->outputNodes = outputNodes;
    nn->trainingSet = trainingSet;
    nn->learningRate = learningRate;

    nn->hiddenLayer = (double*)malloc(nn->hiddenNodes * sizeof(double));
    nn->outputLayer = (double*)malloc(nn->outputNodes * sizeof(double));

    nn->hiddenLayerBias = (double*)malloc(nn->hiddenNodes * sizeof(double));
    nn->outputLayerBias = (double*)malloc(nn->outputNodes * sizeof(double));
    
    nn->hiddenWeights = (double**)malloc(nn->inputNodes * sizeof(double*));
    for (int i = 0; i < nn->inputNodes; i++) {
        nn->hiddenWeights[i] = (double*)malloc(nn->hiddenNodes * sizeof(double));
    }

    nn->outputWeights = (double**)malloc(nn->hiddenNodes * sizeof(double*));
    for (int i = 0; i < nn->hiddenNodes; i++) {
        nn->outputWeights[i] = (double*)malloc(nn->outputNodes * sizeof(double));
    }

    for (int i = 0; i < nn->inputNodes; i++) {
        for (int j = 0; j < nn->hiddenNodes; j++) {
            nn->hiddenWeights[i][j] = randomWeight();
        }
    }

    for (int i = 0; i < nn->hiddenNodes; i++) {
        for (int j = 0; j < nn->outputNodes; j++) {
            nn->outputWeights[i][j] = randomWeight();
        }
    }

    for (int i = 0; i < nn->hiddenNodes; i++) {
        nn->hiddenLayerBias[i] = randomWeight();
    }

    for (int i = 0; i < nn->outputNodes; i++) {
        nn->outputLayerBias[i] = randomWeight();
    }

    return nn;
}

void trainNeuralNetwork(NeuralNetwork* nn, double** inputs, double** targets) {
    int* trainingSetOrder = malloc(nn->trainingSet * sizeof(int));
    for (int i = 0; i < nn->trainingSet; i++) {
        trainingSetOrder[i] = i;
    }
    int epochs = 10000;

    for (int epoch = 0; epoch < epochs; epoch++) {
        shuffle(trainingSetOrder, nn->trainingSet);
        for (int i = 0; i < nn->trainingSet; i++) {
            forwardPropagation(nn, inputs[trainingSetOrder[i]]);
            backPropagation(nn, inputs[trainingSetOrder[i]], targets[trainingSetOrder[i]]);
        }

        double error = 0.0f;
        for (int j = 0; j < nn->outputNodes; j++) {
            error += (targets[nn->trainingSet-1][j] - nn->outputLayer[j]) * sigmoidDerivative(nn->outputLayer[j]);
        }
        printf("Loss: %g\n", error);
    }

    free(trainingSetOrder);
}

void forwardPropagation(NeuralNetwork* nn, double* inputs) {
    for (int j = 0; j < nn->hiddenNodes; j++) {
        double activation = nn->hiddenLayerBias[j];

        for (int k = 0; k < nn->inputNodes; k++) {
            activation += inputs[k] * nn->hiddenWeights[k][j];
        }

        nn->hiddenLayer[j] = sigmoid(activation);
    }

    for (int j = 0; j < nn->outputNodes; j++) {
        double activation = nn->outputLayerBias[j];

        for (int k = 0; k < nn->hiddenNodes; k++) {
            activation += nn->hiddenLayer[k] * nn->outputWeights[k][j];
        }

        nn->outputLayer[j] = sigmoid(activation);
    }
}

void backPropagation(NeuralNetwork* nn, double* inputs, double* targets) {
    double* deltaOutput = malloc(nn->outputNodes * sizeof(double));
        
    for (int j = 0; j < nn->outputNodes; j++) {
        double error = targets[j] - nn->outputLayer[j];
        deltaOutput[j] = error * sigmoidDerivative(nn->outputLayer[j]);
    }

    double* deltaHidden = malloc(nn->hiddenNodes * sizeof(double));

    for (int j = 0; j < nn->hiddenNodes; j++) {
        double error = 0.0f;

        for (int k = 0; k < nn->outputNodes; k++) {
            error += deltaOutput[k] * nn->outputWeights[j][k];
        }

        deltaHidden[j] = error * sigmoidDerivative(nn->hiddenLayer[j]);
    }

    for (int j=0;j<nn->outputNodes;j++) {
        nn->outputLayerBias[j] += deltaOutput[j] * nn->learningRate;
        for (int k=0;k<nn->hiddenNodes;k++) {
            nn->outputWeights[k][j] += nn->hiddenLayer[k] * deltaOutput[j] * nn->learningRate;
        }
    }

    for (int j=0;j<nn->hiddenNodes;j++) {
        nn->hiddenLayerBias[j] += deltaHidden[j] * nn->learningRate;
        for (int k=0;k<nn->inputNodes;k++) {
            nn->hiddenWeights[k][j] += inputs[k] * deltaHidden[j] * nn->learningRate;
        }
    }

    free(deltaOutput);
    free(deltaHidden);
}

void finalPrediction(NeuralNetwork* nn, double* inputs, double* targets) {
    forwardPropagation(nn, inputs);
    printf("\nFinal prediction\nInput :");
    for (int i = 0; i < nn->inputNodes; i++) {
        printf(" %g", inputs[i]);
    }
    printf("\nTarget :");
    for (int i = 0; i < nn->outputNodes; i++) {
        printf(" %g", targets[i]);
    }
    printf("\nPredicted Output :");
    for (int i = 0; i < nn->outputNodes; i++) {
        printf(" %g", nn->outputLayer[i]);
    }
    printf("\n");
}

void freeNeuralNetwork(NeuralNetwork* nn) {
    free(nn->hiddenLayer);
    free(nn->outputLayer);
    free(nn->hiddenLayerBias);
    free(nn->outputLayerBias);

    for (int i = 0; i < nn->inputNodes; i++) {
        free(nn->hiddenWeights[i]);
    }
    free(nn->hiddenWeights);

    for (int i = 0; i < nn->hiddenNodes; i++) {
        free(nn->outputWeights[i]);
    }
    free(nn->outputWeights);

    free(nn);
}
