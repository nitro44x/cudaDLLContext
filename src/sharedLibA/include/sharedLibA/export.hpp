#pragma once

#ifdef sharedLibA_EXPORTS
#define API __declspec(dllexport)
#else
#define API __declspec(dllimport)
#endif