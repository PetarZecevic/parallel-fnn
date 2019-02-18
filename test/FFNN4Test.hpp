#ifndef FFNN4TEST_HPP
#define FFNN4TEST_HPP

#include <cppunit/extensions/HelperMacros.h>
#include "FFNN4.hpp"

class FFNN4Test : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(FFNN4Test);
    CPPUNIT_TEST_EXCEPTION(testConstructor, std::bad_alloc);
    CPPUNIT_TEST(testBadInput);
    CPPUNIT_TEST(testOutputIdentity);
    CPPUNIT_TEST(testOutputSigmoid);
    CPPUNIT_TEST(testOutputTanh);
    CPPUNIT_TEST(testOutputArctan);
    CPPUNIT_TEST(testOutputBinary);
    CPPUNIT_TEST_SUITE_END();
protected:
    std::vector<double> octaveInput;
    std::vector<std::vector<double>> octaveMatrix1;
    std::vector<std::vector<double>> octaveMatrix2;
    std::vector<std::vector<double>> octaveMatrix3;
    std::vector<double> octaveSigmoidOutput;
    std::vector<double> octaveTanhOutput;
    std::vector<double> octaveArctanOutput;
    std::vector<double> octaveBinaryOutput;
public:
    void setUp();
    void tearDown();
protected:
    void testConstructor();
    void testBadInput();
    void testOutputIdentity();
    void testOutputSigmoid();
    void testOutputTanh();
    void testOutputArctan();
    void testOutputBinary();
};

#endif