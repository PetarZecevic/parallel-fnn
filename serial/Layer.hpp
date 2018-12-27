#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>
#include "Activation.hpp"

typedef enum {INPUT , HIDDEN, OUTPUT}LayerType;
typedef enum {SIGMOID, IDENTITY, TANH, ARCTAN, BINARY}ActivationFunctionType;

/*
Class that models one layer in Neural Network.
Layer has input, weight matrix, activation function and output.
*/
class Layer
{
	unsigned int neuronsInPreviousLayer;
	unsigned int neuronsInLayer;
    std::vector<float> input;
    std::vector<float> output;
    std::vector< std::vector <float> > weightMatrix;
    LayerType layerType;
    ActivationFunction* activation;
public:
    Layer(unsigned int nLayer, unsigned int nPrev, LayerType lType, ActivationFunctionType act);
    
    // Copy given input into layers input.
    void setInput(const std::vector<float>& in);
    
    // Copy given weight matrix into layers weight matrix.
    void setWeightMatrix(const std::vector< std::vector<float> >& vM);
    
    void calculateOutput(void);
    
    // Calculates and returns Layer's output for given input.
    std::vector<float> calculateOutput(const std::vector<float>& in);
    
    std::vector<float> getOutput(void) const;
    
    // Release memory used by activation function pointer.
    ~Layer();
};

#endif // LAYER_HPP
