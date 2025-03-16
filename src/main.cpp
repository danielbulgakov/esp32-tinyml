#include <Arduino.h>
#include "esp_heap_caps.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "ml/model.h"
#include "ml/data.h"

// ========================== PSRAM INIT ======================================

bool initPSRam() {
    log_d("Heap size: %lu", ESP.getHeapSize());
    log_d("Free heap size: %lu", ESP.getFreeHeap());
    if (!psramFound()) {
        log_e("PSRam was not found on init");
        return false;
    }
    if (!psramInit()) {
        log_e("PSRam was not inited successfully");
        return false;
    }
    return true;
}

void printPSRamUsage() {
    // TODO: Maybe check if PSRAM was init or found?
    multi_heap_info_t info;
    heap_caps_get_info(&info, MALLOC_CAP_SPIRAM);
    log_d("Total free bytes: %zu", info.total_free_bytes);
    log_d("Total allocated bytes: %zu", info.total_allocated_bytes);
    log_d("Largest free block: %zu", info.largest_free_block);
    log_d("Minimum free bytes: %zu", info.minimum_free_bytes);
    log_d("Allocated blocks: %zu", info.allocated_blocks);
    log_d("Free blocks: %zu", info.free_blocks);
    log_d("Total blocks: %zu", info.total_blocks);
}

struct psdestructor {
    void operator()(void* x) { free(x); }
};

typedef std::unique_ptr<uint8_t, psdestructor> UPSRam;

UPSRam unique_psmalloc(size_t size) {
    auto smart_ptr = UPSRam(static_cast<uint8_t*>(ps_malloc(size)));
    if (!smart_ptr) {
        log_e("Error allocating memory in psram size(bytes): %llu", size);
    }
    return smart_ptr;
}

// ========================== TINYML ==========================================

constexpr size_t IMAGE_SIZE = (28*28); // number of pixels per image (grayscale)
constexpr size_t MODEL_SIZE = MNIST_model_data_len; // model stored in bytes

// Some tinyml specific consts
constexpr size_t TENSOR_SIZE = 100 * 1024; // 1024 KB for tensors

// Make as global vars so we can access them from both setup() and loop() funcs
tflite::MicroInterpreter* interpreter;
tflite::MicroErrorReporter micro_error_reporter;

UPSRam image_mem;
typedef float ImageDataType;

void setup_tinyml_model() {
    // Allocate mem for needs
    UPSRam model_mem  = unique_psmalloc(MODEL_SIZE);
    UPSRam tensor_mem = unique_psmalloc(TENSOR_SIZE);

    // Get the model version
    const tflite::Model* model = tflite::GetModel(MNIST_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        log_e("Incorrect version of TFLite model");
        return;
    }

    // Init operation resolver class
    static tflite::AllOpsResolver resolver;

    // Create the interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_mem.get(), TENSOR_SIZE, &micro_error_reporter);
    interpreter = &static_interpreter;

    // TfLiteTensor* input_tensor = interpreter->input(0);
    // log_i("Input tensor type: %d\n", input_tensor->type);

    // Tring to allocate tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        log_e("Can not allocate tensors");
        return;
    }

    log_i("TinyML successfully started!");
}

// ========================== ARDUINO TESTS ===================================

void test_psram() {
    log_i("Starting ESP32-S3 PSRAM Test");
    initPSRam();
    printPSRamUsage();
    auto p = unique_psmalloc(1000);
    printPSRamUsage();
    p.get()[0] = 'a';
    log_i("%c", static_cast<char>(p.get()[0]));
}

// ========================== ARDUINO SPECIFIC FUNC ===========================

void setup() {
    Serial.begin(115200);
    setup_tinyml_model();
    image_mem = unique_psmalloc(IMAGE_SIZE * sizeof(float));
}

void loop() {
    // Simple image load to a model
    ImageDataType* input = interpreter->input(0)->data.f;
    for (size_t i = 0; i < IMAGE_SIZE; i++) {
        input[i] = image_mem.get()[i];
    }

    // Start the model
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Invoke failed!");
        return;
    }

    // Extract the results
    ImageDataType* output = interpreter->output(0)->data.f;

    // Find classified class
    uint8_t label;
    float argmax = -1;

    for (size_t i = 0; i < 10; i++) {
        Serial.printf("Class[%d] = %.2f %%\n", i, output[i]);
        if (output[i] > argmax) {
            argmax = output[i];
            label = i;
        }
    }

    Serial.printf("The predicted number is: %d (%.2f %%)\n", label, argmax);
    delay(5000);
}
