#pragma once

#ifdef WIN32
#ifdef sharedLibA_EXPORTS
#define API __declspec(dllexport)
#else
#define API __declspec(dllimport)
#endif
#else
#define API 
#endif