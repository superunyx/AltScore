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
import java.nio.channels.FileChannel
import java.nio.FloatBuffer
import kotlin.concurrent.thread
import kotlin.math.sqrt

class TFLiteModule(reactContext: ReactApplicationContext) : ReactContextBaseJavaModule(reactContext) {

    private val TAG = "TFLiteModule"

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
                options.setNumThreads(1) // Avoid uncontrolled thread spawning
                interpreter = Interpreter(mappedByteBuffer, options)
                emitLog("Interpreter initialized with 1 thread and FlexDelegate")

                // 3. Prepare dummy data matching expected shape [1, 30, 3] and [1, 1]
                val numWindows = 1
                val days = 30
                val features = 3
                
                val xBuffer = FloatBuffer.allocate(numWindows * days * features)
                for (i in 0 until numWindows * days * features) { xBuffer.put(0.5f) }
                
                val yBuffer = FloatBuffer.allocate(numWindows * 1)
                for (i in 0 until numWindows) { yBuffer.put(0.5f) }
                
                val trainInputs: MutableMap<String, Any> = HashMap()
                val trainOutputs: MutableMap<String, Any> = HashMap()
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

                // 5. Compute the delta using export_weights
                emitStatus("Exporting Weights")
                emitLog("Training finished, exporting weights")
                val exportInputs = mapOf<String, Any>("dummy" to FloatBuffer.allocate(1))
                val exportOutputs: MutableMap<String, Any> = HashMap()
                
                val outputNames = interpreter.getSignatureOutputs("export_weights")
                for (name in outputNames) {
                    val tensor = interpreter.getOutputTensorFromSignature(name, "export_weights")
                    exportOutputs[name] = FloatBuffer.allocate(tensor.numElements())
                }
                
                interpreter.runSignature(exportInputs, exportOutputs, "export_weights")
                emitLog("Exported ${outputNames.size} weight tensors successfully")
                
                var totalBytes = 0
                var sumOfSquares = 0.0
                for (name in outputNames) {
                    val buffer = exportOutputs[name] as FloatBuffer
                    buffer.rewind()
                    totalBytes += buffer.capacity() * 4
                    while (buffer.hasRemaining()) {
                        val v = buffer.get().toDouble()
                        sumOfSquares += v * v
                    }
                }
                val l2Norm = sqrt(sumOfSquares)
                
                val duration = System.currentTimeMillis() - startTime
                emitLog("Round complete in ${duration}ms")
                emitStatus("Done")
                
                val result = WritableNativeMap()
                result.putString("status", "success")
                result.putString("message", "export_weights signature is wired and FloatBuffers allocated for ${outputNames.size} tensors.")
                result.putInt("duration", duration.toInt())
                result.putInt("byteSize", totalBytes)
                result.putDouble("l2Norm", l2Norm)
                
                emitLog("Resolving promise")
                promise.resolve(result)
                
            } catch (e: Throwable) {
                emitStatus("Error")
                emitLog("Exception or Error caught in background thread: ${e.message}")
                promise.reject("TFLITE_ERROR", "Failed to run training: ${e.message}", e)
            } finally {
                // Ensure native resources are properly freed regardless of exceptions
                interpreter?.close()
                flexDelegate?.close()
                emitLog("Native resources freed")
            }
        }
    }
    
    // Required for React Native events
    @ReactMethod
    fun addListener(eventName: String) {}

    @ReactMethod
    fun removeListeners(count: Int) {}
}
