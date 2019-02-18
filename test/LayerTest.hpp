#ifndef LAYERTEST_HPP
#define LAYERTEST_HPP

#include <cppunit/extensions/HelperMacros.h>
#include "Layer.hpp"

class LayerTest : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(LayerTest);
    CPPUNIT_TEST_EXCEPTION(testConstructor, std::bad_alloc);
    CPPUNIT_TEST(testSetWeight);
    CPPUNIT_TEST(testInputLayer);
    CPPUNIT_TEST(testHiddenLayer);
    CPPUNIT_TEST(testOutputLayer);
    CPPUNIT_TEST_SUITE_END();
protected:
    std::vector<double> octaveInput1, octaveInput2;
    std::vector<std::vector<double>> octaveMatrix1, octaveMatrix2;
    std::vector<double> octaveOutput1, octaveOutput2;
public:
    void setUp();
    void tearDown();
protected:
    void testConstructor();
    void testSetWeight();
    void testInputLayer();
    void testHiddenLayer();
    void testOutputLayer();
};

#endif