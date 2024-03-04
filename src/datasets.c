#include "headers/includes.h"
#include "headers/datasets.h"
#include "headers/nn.h"

void getXorNum(int* trainingSet, int* inputsLength, int* targetsLength) {
    *inputsLength = 2;
    *targetsLength = 1;
    *trainingSet = 4;
}

NeuralNetwork* getXorDataset(double* inputs[], double* targets[], double* predictInput, double* predictTarget) {
    inputs[0][0] = 0.0f; inputs[0][1] = 0.0f; targets[0][0] = 0.0f;
    inputs[1][0] = 0.0f; inputs[1][1] = 1.0f; targets[1][0] = 1.0f;
    inputs[2][0] = 1.0f; inputs[2][1] = 0.0f; targets[2][0] = 1.0f;
    inputs[3][0] = 1.0f; inputs[3][1] = 1.0f; targets[3][0] = 0.0f;

    predictInput[0] = 1.0f; predictInput[1] = 1.0f;
    predictTarget[0] = 0.0f;

    NeuralNetwork* nn = createNeuralNetwork(2, 2, 1, 4, 0.1f);
    return nn;
}

void getDataOneOutputNum(int* trainingSet, int* inputsLength, int* targetsLength) {
    *inputsLength = 2;
    *targetsLength = 1;
    *trainingSet = 8;
}

NeuralNetwork* getDataOneOutputDataset(double* inputs[], double* targets[], double* predictInput, double* predictTarget) {
    inputs[0][0] = 3.0f; inputs[0][1] = 1.5f; targets[0][0] = 1.0f;
    inputs[1][0] = 2.0f; inputs[1][1] = 1.0f; targets[1][0] = 0.0f;
    inputs[2][0] = 4.0f; inputs[2][1] = 1.5f; targets[2][0] = 1.0f;
    inputs[3][0] = 3.0f; inputs[3][1] = 1.0f; targets[3][0] = 0.0f;
    inputs[4][0] = 3.5f; inputs[4][1] = 0.5f; targets[4][0] = 1.0f;
    inputs[5][0] = 2.0f; inputs[5][1] = 0.5f; targets[5][0] = 0.0f;
    inputs[6][0] = 5.5f; inputs[6][1] = 1.0f; targets[6][0] = 1.0f;
    inputs[7][0] = 1.0f; inputs[7][1] = 1.5f; targets[7][0] = 0.0f;

    predictInput[0] = 4.0f; predictInput[1] = 1.5f;
    predictTarget[0] = 1.0f;

    NeuralNetwork* nn = createNeuralNetwork(2, 3, 1, 8, 0.1f);
    return nn;
}

void getDataTwoOutputNum(int* trainingSet, int* inputsLength, int* targetsLength) {
    *inputsLength = 2;
    *targetsLength = 2;
    *trainingSet = 8;
}

NeuralNetwork* getDataTwoOutputsDataset(double* inputs[], double* targets[], double* predictInput, double* predictTarget) {
    inputs[0][0] = 3.0f; inputs[0][1] = 1.5f; targets[0][0] = 0.0f; targets[0][1] = 1.0f;
    inputs[1][0] = 2.0f; inputs[1][1] = 1.0f; targets[1][0] = 1.0f; targets[1][1] = 0.0f;
    inputs[2][0] = 4.0f; inputs[2][1] = 1.5f; targets[2][0] = 0.0f; targets[2][1] = 1.0f;
    inputs[3][0] = 3.0f; inputs[3][1] = 1.0f; targets[3][0] = 1.0f; targets[3][1] = 0.0f;
    inputs[4][0] = 3.5f; inputs[4][1] = 0.5f; targets[4][0] = 0.0f; targets[4][1] = 1.0f;
    inputs[5][0] = 2.0f; inputs[5][1] = 0.5f; targets[5][0] = 1.0f; targets[5][1] = 0.0f;
    inputs[6][0] = 5.5f; inputs[6][1] = 1.0f; targets[6][0] = 0.0f; targets[6][1] = 1.0f;
    inputs[7][0] = 1.0f; inputs[7][1] = 1.5f; targets[7][0] = 1.0f; targets[7][1] = 0.0f;

    predictInput[0] = 4.0f; predictInput[1] = 1.5f;
    predictTarget[0] = 0.0f; predictTarget[1] = 1.0f;

    NeuralNetwork* nn = createNeuralNetwork(2, 3, 2, 8, 0.1f);
    return nn;
}
