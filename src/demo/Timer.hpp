/*
 * GraphBLAS Template Library, Version 2.0
 *
 * Copyright 2019 Carnegie Mellon University, Battelle Memorial Institute, and
 * Authors. All Rights Reserved.
 *
 * THIS MATERIAL WAS PREPARED AS AN ACCOUNT OF WORK SPONSORED BY AN AGENCY OF
 * THE UNITED STATES GOVERNMENT.  NEITHER THE UNITED STATES GOVERNMENT NOR THE
 * UNITED STATES DEPARTMENT OF ENERGY, NOR THE UNITED STATES DEPARTMENT OF
 * DEFENSE, NOR CARNEGIE MELLON UNIVERSITY, NOR BATTELLE, NOR ANY OF THEIR
 * EMPLOYEES, NOR ANY JURISDICTION OR ORGANIZATION THAT HAS COOPERATED IN THE
 * DEVELOPMENT OF THESE MATERIALS, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
 * ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS,
 * OR USEFULNESS OR ANY INFORMATION, APPARATUS, PRODUCT, SOFTWARE, OR PROCESS
 * DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED
 * RIGHTS..
 *
 * Released under a BSD (SEI)-style license, please see license.txt or contact
 * permission@sei.cmu.edu for full terms.
 *
 * DM18-0559
 */

#ifndef __TIMER_HPP
#define __TIMER_HPP

#include <chrono>

//****************************************************************************
template <class ClockT = std::chrono::system_clock>
class Timer
{
public:
    typedef std::chrono::time_point<ClockT> TimeType;

    Timer() : start_time(), stop_time() {}

    TimeType start() { return (start_time = ClockT::now()); }
    TimeType stop()  { return (stop_time  = ClockT::now()); }

    double elapsed() const
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            stop_time - start_time).count();
    }

private:
    TimeType start_time, stop_time;
};

#endif
