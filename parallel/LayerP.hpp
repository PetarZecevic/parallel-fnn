#ifndef LAYERP_HPP
#define LAYERP_HPP

#include <vector>
#include "ActivationP.hpp"
#include "tbb/task.h"

typedef enum{INPUTP , HIDDENP, OUTPUTP}LayerTypeP;
typedef enum {SIGMOIDP, IDENTITYP, TANHP, ARCTANP, BINARYP}ActivationFunctionTypeP;



class LTaskContinuation : public tbb::task
{
	tbb::task* execute()
	{
		return NULL;
	}
};

class LTask : public tbb::task
{	
	const std::vector<double>& input;
	const std::vector<std::vector <double> >& weightMatrix;
	std::vector<double>& output;
	unsigned int rowBegin;
	unsigned int rowEnd;
	unsigned int columns;
	ActivationFunction* actFun;
	bool isInLayer;
public:
	LTask(const std::vector<double>& in, const std::vector< std::vector<double> >& w, std::vector<double>& out, unsigned int begin, unsigned int end, unsigned int col, ActivationFunction* fun, bool inLayer) : 
		input(in),
		weightMatrix(w),
		output(out),
		rowBegin(begin),
		rowEnd(end),
		columns(col),
		actFun(fun),
		isInLayer(inLayer)
	{}
	
	void performSerial();
	tbb::task* execute();
};

/*
Class that models one layer in Neural Network.
Layer has input, weight matrix, activation function and output.
*/
class LayerP
{
	unsigned int neuronsInPreviousLayer;
	unsigned int neuronsInLayer;
    std::vector<double> input;
    std::vector<double> output;
    std::vector< std::vector <double> > weightMatrix;
    LayerTypeP layerType;
    ActivationFunction* activation;
    void action(const std::vector<double>& in, std::vector<double>& out);
public:
    LayerP(unsigned int nPrev, unsigned int nLayer, LayerTypeP lType, ActivationFunctionTypeP act);
    
    // Copy given input into layers input.
    void setInput(const std::vector<double>& in);
    
    // Copy given weight matrix into layers weight matrix.
    void setWeightMatrix(const std::vector< std::vector<double> >& vM);
    
    std::vector<std::vector<double>> getWeightMatrix(void);

    void calculateOutput(void);
    
    // Calculates and returns Layer's output for given input.
    std::vector<double> calculateOutput(const std::vector<double>& in);
    
    std::vector<double> getOutput(void) const;
    
    unsigned int getNeuronsNumLayer() const;
    unsigned int getNeuronsNumPrevLayer() const;

    // Release memory used by activation function pointer.
    ~LayerP();
};

#endif // LAYERP_HPP
