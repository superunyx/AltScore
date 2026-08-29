package com.anonymous.expotemplateblank

import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.WritableNativeMap
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.flex.FlexDelegate
import java.io.FileInputStream
import java.nio.channels.FileChannel
import java.io.File
import java.nio.FloatBuffer
import kotlin.concurrent.thread

class TFLiteModule(reactContext: ReactApplicationContext) : ReactContextBaseJavaModule(reactContext) {

    override fun getName(): String {
        return "TFLiteModule"
    }

    @ReactMethod
    fun runLocalTrainingRound(promise: Promise) {
        // Move to a dedicated background thread so we don't block the React Native bridge/UI
        thread(start = true) {
            try {
                // 1. Load the model from assets
                val assetManager = reactApplicationContext.assets
                val assetFileDescriptor = assetManager.openFd("base_model.tflite")
                val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
                val fileChannel = fileInputStream.channel
                val startOffset = assetFileDescriptor.startOffset
                val declaredLength = assetFileDescriptor.declaredLength
                val mappedByteBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)

                // 2. Initialize Interpreter with FlexDelegate for Select TF Ops
                val options = Interpreter.Options()
                options.addDelegate(FlexDelegate())
                val interpreter = Interpreter(mappedByteBuffer, options)

                // 3. Prepare dummy data matching expected shape [1, 30, 3] and [1, 1]
                val numWindows = 1
                val days = 30
                val features = 3
                
                // Using flat FloatBuffers exactly matches the memory footprint
                // without reflection overhead of nested Kotlin arrays
                val xBuffer = FloatBuffer.allocate(numWindows * days * features)
                for (i in 0 until numWindows * days * features) {
                    xBuffer.put(0.5f)
                }
                
                val yBuffer = FloatBuffer.allocate(numWindows * 1)
                for (i in 0 until numWindows) {
                    yBuffer.put(0.5f)
                }
                
                val trainInputs: MutableMap<String, Any> = HashMap()
                val trainOutputs: MutableMap<String, Any> = HashMap()
                val lossBuffer = FloatBuffer.allocate(1)
                
                // 4. Run the "train" signature in a loop (epochs)
                val epochs = 5
                for (epoch in 1..epochs) {
                    xBuffer.rewind()
                    yBuffer.rewind()
                    lossBuffer.rewind()
                    
                    trainInputs["x"] = xBuffer
                    trainInputs["y"] = yBuffer
                    trainOutputs["loss"] = lossBuffer
                    
                    interpreter.runSignature(trainInputs, trainOutputs, "train")
                }

                // 5. Compute the delta using export_weights
                val exportInputs = mapOf<String, Any>("dummy" to FloatBuffer.allocate(1))
                val exportOutputs: MutableMap<String, Any> = HashMap()
                
                val outputNames = interpreter.getSignatureOutputs("export_weights")
                for (name in outputNames) {
                    val tensor = interpreter.getOutputTensorFromSignature(name, "export_weights")
                    exportOutputs[name] = FloatBuffer.allocate(tensor.numElements())
                }
                
                interpreter.runSignature(exportInputs, exportOutputs, "export_weights")
                
                interpreter.close()
                
                val result = WritableNativeMap()
                result.putString("status", "success")
                result.putString("message", "export_weights signature is wired and FloatBuffers allocated for ${outputNames.size} tensors.")
                
                promise.resolve(result)
                
            } catch (e: Exception) {
                promise.reject("TFLITE_ERROR", "Failed to run training: ${e.message}", e)
            }
        }
    }
}
