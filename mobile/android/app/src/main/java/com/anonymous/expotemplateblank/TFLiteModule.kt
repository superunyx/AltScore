package com.anonymous.expotemplateblank

import android.util.Log
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.WritableNativeMap
import com.facebook.react.modules.core.DeviceEventManagerModule
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.flex.FlexDelegate
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.File
import java.nio.channels.FileChannel
import java.nio.FloatBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.concurrent.thread
import kotlin.math.sqrt
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException

class TFLiteModule(reactContext: ReactApplicationContext) : ReactContextBaseJavaModule(reactContext) {

    private val TAG = "TFLiteModule"
    private val client = OkHttpClient()

    override fun getName(): String {
        return "TFLiteModule"
    }
    
    private fun emitLog(message: String) {
        Log.d(TAG, message)
        if (reactApplicationContext.hasActiveCatalystInstance()) {
            reactApplicationContext
                .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
                .emit("TFLiteLog", message)
        }
    }
    
    private fun emitStatus(status: String) {
        if (reactApplicationContext.hasActiveCatalystInstance()) {
            reactApplicationContext
                .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
                .emit("TFLiteStatus", status)
        }
    }

    private fun buildNestedJsonArray(flatArray: FloatArray, shape: IntArray, offset: IntArray, dimension: Int): Any {
        if (dimension == shape.size - 1) {
            val jsonArray = JSONArray()
            val size = shape[dimension]
            for (i in 0 until size) {
                jsonArray.put(flatArray[offset[0]++].toDouble())
            }
            return jsonArray
        } else {
            val jsonArray = JSONArray()
            val size = shape[dimension]
            for (i in 0 until size) {
                jsonArray.put(buildNestedJsonArray(flatArray, shape, offset, dimension + 1))
            }
            return jsonArray
        }
    }

    @ReactMethod
    fun runLocalTrainingRound(promise: Promise) {
        val startTime = System.currentTimeMillis()
        emitLog("runLocalTrainingRound called, starting background thread")
        emitStatus("Idle")
        
        thread(start = true) {
            emitLog("Background thread started")
            emitStatus("Loading Model")
            var interpreter: Interpreter? = null
            var flexDelegate: FlexDelegate? = null
            try {
                // 1. Load the model from assets
                val assetManager = reactApplicationContext.assets
                val assetFileDescriptor = assetManager.openFd("base_model.tflite")
                
                val mappedByteBuffer = FileInputStream(assetFileDescriptor.fileDescriptor).use { fileInputStream ->
                    val fileChannel = fileInputStream.channel
                    fileChannel.map(FileChannel.MapMode.READ_ONLY, assetFileDescriptor.startOffset, assetFileDescriptor.declaredLength)
                }
                emitLog("Model loaded from assets")

                // 2. Initialize Interpreter with FlexDelegate for Select TF Ops
                val options = Interpreter.Options()
                flexDelegate = FlexDelegate()
                options.addDelegate(flexDelegate)
                options.setNumThreads(1)
                interpreter = Interpreter(mappedByteBuffer, options)
                emitLog("Interpreter initialized with 1 thread and FlexDelegate")

                // Pre-extract initial weights
                emitStatus("Exporting Init Weights")
                val exportInputs = mapOf<String, Any>("dummy" to FloatBuffer.allocate(1))
                val outputNames = interpreter.getSignatureOutputs("export_weights")
                
                val initOutputs: MutableMap<String, Any> = java.util.HashMap()
                for (name in outputNames) {
                    val tensor = interpreter.getOutputTensorFromSignature(name, "export_weights")
                    initOutputs[name] = FloatBuffer.allocate(tensor.numElements())
                }
                interpreter.runSignature(exportInputs, initOutputs, "export_weights")
                
                val initWeightsMap = java.util.HashMap<String, FloatArray>()
                for (name in outputNames) {
                    val buffer = initOutputs[name] as FloatBuffer
                    buffer.rewind()
                    val arr = FloatArray(buffer.capacity())
                    buffer.get(arr)
                    initWeightsMap[name] = arr
                }

                // 3. Prepare dummy data matching expected shape [1, 30, 3] and [1, 1]
                val numWindows = 1
                val days = 30
                val features = 3
                
                val xBuffer = FloatBuffer.allocate(numWindows * days * features)
                for (i in 0 until numWindows * days * features) { xBuffer.put(0.5f) }
                
                val yBuffer = FloatBuffer.allocate(numWindows * 1)
                for (i in 0 until numWindows) { yBuffer.put(0.5f) }
                
                val trainInputs: MutableMap<String, Any> = java.util.HashMap()
                val trainOutputs: MutableMap<String, Any> = java.util.HashMap()
                val lossBuffer = FloatBuffer.allocate(1)
                
                // 4. Run the "train" signature in a loop (epochs)
                val epochs = 5
                emitLog("Starting training loop for $epochs epochs")
                for (epoch in 1..epochs) {
                    emitStatus("Epoch $epoch of $epochs")
                    emitLog("Running epoch $epoch")
                    xBuffer.rewind()
                    yBuffer.rewind()
                    lossBuffer.rewind()
                    
                    trainInputs["x"] = xBuffer
                    trainInputs["y"] = yBuffer
                    trainOutputs["loss"] = lossBuffer
                    
                    interpreter.runSignature(trainInputs, trainOutputs, "train")
                    emitLog("Epoch $epoch finished")
                }

                // 4b. Run the "infer" signature to get the new predicted score
                emitLog("Running infer to evaluate learned score")
                xBuffer.rewind()
                val inferInputs = mapOf<String, Any>("x" to xBuffer)
                val inferOutputs = mapOf<String, Any>("output" to FloatBuffer.allocate(1))
                interpreter.runSignature(inferInputs, inferOutputs, "infer")
                
                val outputBuffer = inferOutputs["output"] as FloatBuffer
                outputBuffer.rewind()
                val rawScore = outputBuffer.get()
                val computedScore = (rawScore * 1000).toInt()
                emitLog("Inferred raw score $rawScore, scaled to $computedScore")

                // 5. Compute the delta using export_weights
                emitStatus("Exporting Tuned Weights")
                val tunedOutputs: MutableMap<String, Any> = java.util.HashMap()
                for (name in outputNames) {
                    val tensor = interpreter.getOutputTensorFromSignature(name, "export_weights")
                    tunedOutputs[name] = FloatBuffer.allocate(tensor.numElements())
                }
                interpreter.runSignature(exportInputs, tunedOutputs, "export_weights")
                
                val weightDeltaJson = JSONObject()
                var sumOfSquares = 0.0
                
                for (name in outputNames) {
                    val buffer = tunedOutputs[name] as FloatBuffer
                    buffer.rewind()
                    val tunedArr = FloatArray(buffer.capacity())
                    buffer.get(tunedArr)
                    
                    val initArr = initWeightsMap[name]!!
                    val deltaArr = FloatArray(tunedArr.size)
                    for (i in tunedArr.indices) {
                        val diff = tunedArr[i] - initArr[i]
                        deltaArr[i] = diff
                        sumOfSquares += diff * diff
                    }
                    
                    val tensor = interpreter.getOutputTensorFromSignature(name, "export_weights")
                    val shape = tensor.shape()
                    val offset = intArrayOf(0)
                    
                    val nestedArray = buildNestedJsonArray(deltaArr, shape, offset, 0)
                    weightDeltaJson.put(name, nestedArray)
                }
                
                val l2Norm = sqrt(sumOfSquares)
                emitLog("Delta JSON created successfully (L2: $l2Norm)")

                // 6. Send OkHttp POST
                emitStatus("POSTing JSON")
                val payloadJson = JSONObject()
                payloadJson.put("client_id", "device_001")
                payloadJson.put("weight_delta", weightDeltaJson)
                payloadJson.put("data_samples", numWindows)

                val jsonStr = payloadJson.toString()
                val body = jsonStr.toRequestBody("application/json; charset=utf-8".toMediaType())
                
                // Using 10.0.2.2 for Android emulator -> host
                val request = Request.Builder()
                    .url("http://127.0.0.1:8000/submit_update")
                    .post(body)
                    .build()
                
                var httpCode = -1
                var httpMessage = ""
                client.newCall(request).execute().use { response ->
                    httpCode = response.code
                    httpMessage = response.body?.string() ?: ""
                    if (!response.isSuccessful) {
                        throw IOException("Unexpected code $response: $httpMessage")
                    }
                }
                emitLog("Server responded: $httpCode $httpMessage")
                
                val duration = System.currentTimeMillis() - startTime
                emitLog("Round complete in ${duration}ms")
                emitStatus("Done")
                
                val result = WritableNativeMap()
                result.putString("status", "success")
                result.putString("message", "OkHttp POST succeeded with HTTP $httpCode: $httpMessage")
                result.putInt("duration", duration.toInt())
                result.putInt("byteSize", jsonStr.toByteArray().size)
                result.putDouble("l2Norm", l2Norm)
                result.putInt("computedScore", computedScore)
                
                emitLog("Resolving promise")
                promise.resolve(result)
                
            } catch (e: Throwable) {
                emitStatus("Error")
                emitLog("Exception or Error caught in background thread: ${e.message}")
                promise.reject("TFLITE_ERROR", "Failed to run training: ${e.message}", e)
            } finally {
                interpreter?.close()
                flexDelegate?.close()
                emitLog("Native resources freed")
            }
        }
    }
    
    @ReactMethod
    fun addListener(eventName: String) {}

    @ReactMethod
    fun removeListeners(count: Int) {}
}
