///////////////////////////////////////////////////////////////////////////
//
// NAME
//  Error.h -- a simple error handling class
//
// DESCRIPTION
//  The CError class is used to throw error messages back to the calling program.
//
// Copyright © Richard Szeliski, 2001.
// See Copyright.h for more details
//
///////////////////////////////////////////////////////////////////////////

namespace std {}
using namespace std;

#include <string.h>
#include <stdio.h>
#include <exception>

struct CError : public exception
{
    CError(const char* msg)                 { strcpy_s(message, msg); }
    CError(const char* fmt, int d)          { sprintf_s(message, fmt, d); }
    CError(const char* fmt, float f)        { sprintf_s(message, fmt, f); }
    CError(const char* fmt, const char *s)  { sprintf_s(message, fmt, s); }
    CError(const char* fmt, const char *s,
            int d)                          {
		sprintf_s(message, fmt, s, d); }
    char message[1024];         // longest allowable message
};
