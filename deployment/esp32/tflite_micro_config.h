#pragma once

// Configure arena size based on your board memory budget
#ifndef TFLM_ARENA_SIZE
#define TFLM_ARENA_SIZE (200 * 1024)
#endif

// Adjust to 1 for int8 quantized models
typedef int8_t micro_data_t;
