# ESP32 deployment (TFLite Micro)

Guide for running MicroSign-Edge on ESP32 boards using TensorFlow Lite Micro.

## 1. Prerequisites
- Arduino IDE or ESP-IDF toolchain installed
- TFLite Micro libraries (`arduino-tensorflowlite` or ESP-IDF TFLM component)
- Python 3 on host for exporting and generating the C array

## 2. Export and convert the model
1. Export an int8 TFLite file:
   ```bash
   python -m microbackbone.evaluation.export_tflite \
     --checkpoint outputs/checkpoints/microsignedge_edge_small_best.pth \
     --config microbackbone/config/model.yaml \
     --input-size 32 \
     --output-dir outputs/export \
     --quantize
   ```
2. Convert the quantized model to a C array:
   ```bash
   xxd -i outputs/export/model_int8.tflite > model_data.h
   ```
3. Copy `model_data.h` next to `esp32_main.cpp`.

## 3. Build and flash
- Copy `esp32_main.cpp` and `tflite_micro_config.h` into your project `src/` folder.
- Adjust `TFLM_ARENA_SIZE` in `tflite_micro_config.h` to fit board RAM (start at 200KB and decrease if memory errors appear).
- Compile and flash using Arduino IDE or `idf.py flash` with your board selected.

## 4. Read latency output
The sketch prints per-inference latency using `micros()` around `interpreter.Invoke()`. Open the serial monitor at 115200 baud to see lines like:
```
Predicted class: <id> | latency: <microseconds> us
```

## 5. Memory and accuracy notes
- Always use int8 quantized models for ESP32 to stay within RAM/flash limits.
- Keep input resolution small (e.g., 32×32) for reliable allocations.
- Map output indices to your dataset labels (CIFAR-10 order when training on CIFAR-10).
