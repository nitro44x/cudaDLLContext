#pragma once

#ifdef sharedLibB_EXPORTS
#define API __declspec(dllexport)
#else
#define API __declspec(dllimport)
#endif
