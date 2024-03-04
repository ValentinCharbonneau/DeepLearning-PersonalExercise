#include "headers/includes.h"
#include "headers/datasets.h"
#include "headers/sigmoid.h"
#include "headers/weight.h"
#include "headers/nn.h"

int main(void) {
    srand(time(NULL));

    int number;
    printf("Choose a model\n 1 - Simple XOR model\n 2 - Probabilistic model with a single output\n 3 - Class model with two outputs\n\n >>> ");
    scanf("%d", &number);
    printf("\n\n");

    int* trainingSet = (int*)malloc(sizeof(int));
    int* inputsLength = (int*)malloc(sizeof(int));
    int* targetsLength = (int*)malloc(sizeof(int));

    if (number == 1) {
        getXorNum(trainingSet, inputsLength, targetsLength);
    }
    else if (number == 2) {
        getDataOneOutputNum(trainingSet, inputsLength, targetsLength);
    }
    else if (number == 3) {
        getDataTwoOutputNum(trainingSet, inputsLength, targetsLength);
    }
    else {
        exit(1);
    }

    double** inputs = (double**)malloc(*trainingSet * sizeof(double*));
    double** targets = (double**)malloc(*trainingSet * sizeof(double*));
    for (int i=0; i<*trainingSet; i++) {
        inputs[i] = (double*)malloc(*inputsLength * sizeof(double));
        targets[i] = (double*)malloc(*targetsLength * sizeof(double));
    }
    double* predictInput = (double*)malloc(*inputsLength * sizeof(double));
    double* predictTarget = (double*)malloc(*targetsLength * sizeof(double));

    NeuralNetwork* nn;
    if (number == 1) {
        nn = getXorDataset(inputs, targets, predictInput, predictTarget);
    }
    else if (number == 2) {
        nn = getDataOneOutputDataset(inputs, targets, predictInput, predictTarget);
    }
    else if (number == 3) {
        nn = getDataTwoOutputsDataset(inputs, targets, predictInput, predictTarget);
    }

    trainNeuralNetwork(nn, inputs, targets);
    finalPrediction(nn, predictInput, predictTarget);

    freeNeuralNetwork(nn);
    for (int i=0; i<*trainingSet; i++) {
        free(inputs[i]);
        free(targets[i]);
    }
    free(inputs);
    free(targets);
    free(trainingSet);
    free(inputsLength);
    free(targetsLength);
    free(predictInput);
    free(predictTarget);
}