// Example ESP32 inference using TFLite Micro
// Drop this into an ESP-IDF or Arduino project and include the exported model.tflite

#include "tflite_micro_config.h"
#include "model_data.h"  // generated from model.tflite with xxd or binary embedding

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include <Arduino.h>

// Arena memory for TFLite Micro
constexpr int kArenaSize = TFLM_ARENA_SIZE;
static uint8_t tensor_arena[kArenaSize];

void setup() {
  Serial.begin(115200);
  while (!Serial) {
  }
  Serial.println("Booting MicroSign-Net inference demo");

  // Map the model
  const tflite::Model *model = tflite::GetModel(g_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    return;
  }

  static tflite::AllOpsResolver resolver;

  // Interpreter
  static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                             kArenaSize);
  TfLiteStatus allocate_status = interpreter.AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return;
  }

  TfLiteTensor *input = interpreter.input(0);
  Serial.printf("Input: %d x %d x %d\n", input->dims->data[1], input->dims->data[2],
                input->dims->data[3]);

  // Fill input with a dummy image or sensor data here
  for (int i = 0; i < input->bytes; ++i) {
    input->data.int8[i] = 0;  // replace with real preprocessed data
  }

  uint32_t start = micros();
  TfLiteStatus invoke_status = interpreter.Invoke();
  uint32_t elapsed = micros() - start;

  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed");
    return;
  }

  const TfLiteTensor *output = interpreter.output(0);
  int predicted = 0;
  int8_t max_val = -128;
  for (int i = 0; i < output->bytes; ++i) {
    if (output->data.int8[i] > max_val) {
      max_val = output->data.int8[i];
      predicted = i;
    }
  }

  Serial.printf("Predicted class: %d | latency: %lu us\n", predicted, elapsed);
}

void loop() {
  // no-op
  delay(1000);
}
