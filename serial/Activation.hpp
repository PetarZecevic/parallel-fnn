#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include <cmath>

// Base abstract class for all activation functions.
class ActivationFunction
{
public:
	virtual float calculate(const float& x) = 0;
};

class Sigmoid : public ActivationFunction
{
public:
	float calculate(const float& x)
	{
		return 1 / (1 + exp(-x));
	}
};

class Identity : public ActivationFunction
{
public:
	float calculate(const float& x)
	{
		return x;
	}
};

class TanH : public ActivationFunction
{
public:
	float calculate(const float& x)
	{
		return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
	}
};

class ArcTan : public ActivationFunction
{
public:
	float calculate(const float& x)
	{
		return atan(-x);
	}
};

class BinaryStep : public ActivationFunction
{
public:
	float calculate(const float& x)
	{
		return (x < 0) ? 0 : 1;
	}
};

#endif // ACTIVATION_HPP
