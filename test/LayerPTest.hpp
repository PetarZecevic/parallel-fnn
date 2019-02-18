#ifndef LAYERPTEST_HPP
#define LAYERPTEST_HPP

#include <cppunit/extensions/HelperMacros.h>
#include "LayerP.hpp"

class LayerPTest : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(LayerPTest);
    CPPUNIT_TEST_EXCEPTION(testConstructor, std::bad_alloc);
    CPPUNIT_TEST(testSetWeight);
    CPPUNIT_TEST(testInputLayer);
    CPPUNIT_TEST(testHiddenLayer);
    CPPUNIT_TEST(testOutputLayer);
    CPPUNIT_TEST_SUITE_END();
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
