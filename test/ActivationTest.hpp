#ifndef ACTIVATIONTEST_HPP
#define ACTIVATIONTEST_HPP

#include <cppunit/extensions/HelperMacros.h>
#include "Activation.hpp"

class ActivationTest : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(ActivationTest);
    CPPUNIT_TEST(testSigmoid);
    CPPUNIT_TEST(testTanh);
    CPPUNIT_TEST(testArctan);
    CPPUNIT_TEST(testBinary);
    CPPUNIT_TEST(testIdentity);
    CPPUNIT_TEST_SUITE_END();
protected:
    ActivationFunction* sigmoid; 
    ActivationFunction* tanH; 
    ActivationFunction* arcTan;
    ActivationFunction* binary;
    ActivationFunction* identity;
public:
    void setUp();
    void tearDown();
protected:
    void testSigmoid();
    void testTanh();
    void testArctan();
    void testBinary();
    void testIdentity();
};

#endif
