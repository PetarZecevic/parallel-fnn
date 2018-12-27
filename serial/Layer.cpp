#include <cmath>
#include "Layer.hpp"

Layer::Layer(unsigned int nPrev, unsigned int nLayer, LayerType lType,  ActivationFunctionType act):
	neuronsInPreviousLayer(nPrev),
	neuronsInLayer(nLayer),
    input(neuronsInPreviousLayer), // Construct input vector as a vector of size neuronsInPreviousLayer.
   	output(neuronsInLayer), // Construct output vector as a vector of size neuronsInLayer.
   	weightMatrix(neuronsInLayer, std::vector<float>(neuronsInPreviousLayer)), // Construct weight matrix as matrix with neuronsInLayer rows and neuronsInPreviousLayer columns. 
   	layerType(lType)
   	{
   		switch(act)
		{
			case SIGMOID:
				activation = new Sigmoid();
				break;
			case IDENTITY:
				activation = new Identity();
				break;
			case TANH:
				activation = new TanH();
				break;
			case ARCTAN:
				activation = new ArcTan();
				break;
			case BINARY:
				activation = new BinaryStep();
				break;
			default:
				activation = new Sigmoid();
				break;
		}
   	}

void Layer::setInput(const std::vector<float>& in)
{
	input = in;
}
    

void Layer::setWeightMatrix(const std::vector< std::vector<float> >& vM)
{
	for(unsigned int i = 0; i < neuronsInLayer; ++i)
	{
		for(unsigned int j = 0; j < neuronsInPreviousLayer; ++j)
		{
			weightMatrix[i][j] = vM[i][j];
		}
	}
}

void Layer::calculateOutput(void)
{
	if(layerType == INPUT)
	{
		// In case of input layer, just apply activation function to input vector.
		for(unsigned int i = 0; i < neuronsInLayer; ++i)
		{
		    //output[i] = 1 / (1 + exp(-input[i]));
		    output[i] = activation->calculate(input[i]);
		}
	}
	else
	{
		// Multiply weigth matrix with input vector.
		for(unsigned int i = 0; i < neuronsInLayer; ++i)
		{
		    output[i] = 0;
		    for(unsigned int j = 0; j < neuronsInPreviousLayer; ++j)
		    {
		        output[i] += weightMatrix[i][j] * input[j];
		    }
		    // Apply activation function.
		    output[i] = activation->calculate(output[i]);
		}
	}
}

std::vector<float> Layer::calculateOutput(const std::vector<float>& in)
{
	// Temporary output.
	std::vector<float> out(neuronsInLayer);
	
	if(layerType == INPUT)
	{
		// In case of input layer, just apply activation function to input vector.
		for(unsigned int i = 0; i < neuronsInLayer; ++i)
		{
		    out[i] = activation->calculate(in[i]);
		}
	}
	else
	{
		// Multiply weigth matrix with input vector.
		for(unsigned int i = 0; i < neuronsInLayer; ++i)
		{
		    out[i] = 0;
		    for(unsigned int j = 0; j < neuronsInPreviousLayer; ++j)
		    {
		        out[i] += weightMatrix[i][j] * in[j];
		    }
		    // Apply activation function.
		    out[i] = activation->calculate(out[i]);
		}
	}
}

std::vector<float> Layer::getOutput(void) const
{
	return output;
}

Layer::~Layer()
{
	delete activation;
}
