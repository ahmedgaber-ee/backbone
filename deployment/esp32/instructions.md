# ESP32 Deployment Guide (TinyML / TFLite Micro)

## 1. Prerequisites
- Install Arduino IDE or ESP-IDF toolchain
- Clone TensorFlow Lite Micro examples (included with `arduino-tensorflowlite` or ESP-IDF component)

## 2. Add the Model
1. Export `model_int8.tflite` using `microbackbone/evaluation/export_tflite.py --quantize`.
2. Convert it to a C array:
```bash
xxd -i model_int8.tflite > model_data.h
```
3. Copy `model_data.h` next to `esp32_main.cpp`.

## 3. Build Steps
- Copy `esp32_main.cpp` and `tflite_micro_config.h` into your project `src/` folder.
- Ensure `TFLM_ARENA_SIZE` fits in RAM (start with 200KB and reduce if needed).
- Select your ESP32 board and flash via Arduino or `idf.py flash`.

## 4. Latency Measurement
The sketch prints latency using `micros()` around `interpreter.Invoke()`. Open the serial monitor at 115200 baud to read:
```
Predicted class: <id> | latency: <microseconds> us
```

## 5. Memory Notes
- Int8 quantized models are required for ESP32.
- Keep input size small (e.g., 32x32) to stay within RAM limits.
- Reduce `TFLM_ARENA_SIZE` if you see memory allocation failures.

## 6. Interpreting Output
- The output tensor is int8 logits; the demo picks the index with the largest value as the predicted class.
- Map indices to your dataset label list (CIFAR-10 order if training on CIFAR-10).
