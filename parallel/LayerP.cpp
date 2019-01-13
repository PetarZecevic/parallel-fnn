#include <cmath>
#include "LayerP.hpp"


class LTaskContinuation : public tbb::task
{
	tbb::task* execute()
	{
		return NULL;
	}
};

// Helper class for Layer output calculation.
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


// Class for parallel multiplication of two vectors used in parallel_reduce template.
// It's assumed that vectors have same number of elements.
class VectorMultiply
{
	const std::vector<double>& my_vec1;
	const std::vector<double>& my_vec2;
public:
	double my_result;
	void operator() (const tbb::blocked_range<size_t>& r)
	{
		double tmp_result = my_result;
		for(size_t i = r.begin(); i != r.end(); ++i)
		{
			tmp_result += (my_vec1[i] * my_vec2[i]);
		}
		my_result = tmp_result;
	}
	VectorMultiply(const std::vector<double>& vec1, const std::vector<double>& vec2):
		my_vec1(vec1),
		my_vec2(vec2),
		my_result(0)
	{}
	VectorMultiply(VectorMultiply& vm, tbb::split) :
		my_vec1(vm.my_vec1),
		my_vec2(vm.my_vec2),
		my_result(0)
	{}
	void join(const VectorMultiply& vm)
	{
		my_result += vm.my_result;
	}
};

void LTask::performSerial()
{
	if(isInLayer)
	{
		for(unsigned int i = rowBegin; i < rowEnd; i++)
		{
			output[i] = actFun->calculate(input[i]);
		}
	}
	else
	{
		for(unsigned int i = rowBegin; i < rowEnd; i++)
		{
			output[i] = 0;
			for(unsigned int j = 0; j < columns; j++)
			{
				output[i] += weightMatrix[i][j] * input[j];
			}
			output[i] = actFun->calculate(output[i]);
		}
	}	
}
	
tbb::task* LTask::execute()
{
	unsigned int rows = rowEnd - rowBegin;
	unsigned int P = rows * columns;
	if(P <= 10000 || rows == 1)
	{	
		if(rows == 1)
		{
			VectorMultiply reductor(weightMatrix[rowBegin], input);
			tbb::parallel_reduce(tbb::blocked_range<size_t>(0, columns, 1000), reductor);
			output[rowBegin] = actFun->calculate(reductor.my_result);
		}
		else
		{
			performSerial();
		}
		return NULL;		
	}
	else
	{
		unsigned int halfInterval = (rowEnd - rowBegin) / 2;
		unsigned int middle = rowBegin + halfInterval;
		
		LTaskContinuation& c = *new(tbb::task::allocate_continuation()) LTaskContinuation();
		//LTask& left = *new(c.allocate_child()) LTask(input, weightMatrix, output, rowBegin, middle, columns, actFun, isInLayer);
		LTask& right = *new(c.allocate_child()) LTask(input, weightMatrix, output, middle, rowEnd, columns, actFun, isInLayer);
		tbb::task::recycle_as_child_of(c);
		rowEnd = middle;
		c.set_ref_count(2);
		spawn(right);
		//return &left;
		return this;
	}					
}

LayerP::LayerP(unsigned int nPrev, unsigned int nLayer, LayerTypeP lType,  ActivationFunctionTypeP act):
	neuronsInPreviousLayer(nPrev),
	neuronsInLayer(nLayer),
    input(neuronsInPreviousLayer), // Construct input vector as a vector of size neuronsInPreviousLayer.
   	output(neuronsInLayer), // Construct output vector as a vector of size neuronsInLayer.
   	weightMatrix(neuronsInLayer, std::vector<double>(neuronsInPreviousLayer)), // Construct weight matrix as matrix with neuronsInLayer rows and neuronsInPreviousLayer columns. 
   	layerType(lType)
{	
		if(lType == INPUTP)
		{
			neuronsInPreviousLayer = neuronsInLayer;
			input.resize(neuronsInLayer);
		}
		switch(act)
		{
			case SIGMOIDP:
				activation = new Sigmoid();
				break;
			case IDENTITYP:
				activation = new Identity();
				break;
			case TANHP:
				activation = new TanH();
				break;
			case ARCTANP:
				activation = new ArcTan();
				break;
			case BINARYP:
				activation = new BinaryStep();
				break;
			default:
				activation = new Sigmoid();
				break;
		}
}

void LayerP::setInput(const std::vector<double>& in)
{
	if(in.size() != neuronsInPreviousLayer)
		return;
	else
		input = in;
}
    

void LayerP::setWeightMatrix(const std::vector< std::vector<double> >& vM)
{
	// If dimensions don't agree return.
	if(vM.size() != neuronsInLayer)
		return;
	for(unsigned int i = 0; i < neuronsInLayer; ++i)
	{
		weightMatrix[i] = vM[i];
	}
}

std::vector<std::vector<double>> LayerP::getWeightMatrix(void)
{
	return weightMatrix;
}

void LayerP::action(const std::vector<double>& in, std::vector<double>& out)
{
	
	if(layerType == INPUTP)
	{
		LTask& root = *new(tbb::task::allocate_root()) LTask(in, weightMatrix, out, 0, neuronsInLayer, 1, activation, true);
		tbb::task::spawn_root_and_wait(root);
	}
	else
	{
		LTask& root = *new(tbb::task::allocate_root()) LTask(in, weightMatrix, out, 0, neuronsInLayer, neuronsInPreviousLayer, activation, false);
		tbb::task::spawn_root_and_wait(root);
	}
}

void LayerP::calculateOutput(void)
{
	action(input, output);
}

std::vector<double> LayerP::calculateOutput(const std::vector<double>& in)
{

	if(in.size() != neuronsInPreviousLayer)
		return std::vector<double>{0};
	// Temporary output.
	std::vector<double> out(neuronsInLayer);
	action(in, out);
	return out;
}

std::vector<double> LayerP::getOutput(void) const
{
	return output;
}

unsigned int LayerP::getNeuronsNumLayer() const
{
	return neuronsInLayer;
}

unsigned int LayerP::getNeuronsNumPrevLayer() const
{
	return neuronsInPreviousLayer;
}

LayerP::~LayerP()
{
	delete activation;
}
