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

class TFLiteModule(reactContext: ReactApplicationContext) : ReactContextBaseJavaModule(reactContext) {

    override fun getName(): String {
        return "TFLiteModule"
    }

    @ReactMethod
    fun runLocalTrainingRound(promise: Promise) {
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

            // 3. Prepare dummy data: shape [13, 30, 3] for x, [13, 1] for y
            // We just create empty arrays for the spike to see if execution succeeds.
            val xTrain = Array(13) { Array(30) { FloatArray(3) { 0.5f } } }
            val yTrain = Array(13) { FloatArray(1) { 0.5f } }
            
            // 4. Run the "train" signature
            val trainInputs: MutableMap<String, Any> = HashMap()
            trainInputs["x"] = xTrain
            trainInputs["y"] = yTrain
            val trainOutputs: MutableMap<String, Any> = HashMap()
            trainOutputs["loss"] = FloatBuffer.allocate(1) // Output is a scalar loss
            
            // In Android TFLite Java API, we use runSignature
            // Note: The loss tensor might just be a FloatArray or FloatBuffer
            interpreter.runSignature(trainInputs, trainOutputs, "train")

            // 5. Compute the delta using export_weights
            val exportInputs = emptyMap<String, Any>()
            val exportOutputs: MutableMap<String, Any> = HashMap()
            
            // Dynamically allocate FloatBuffers for all exported weight tensors
            val outputNames = interpreter.getSignatureOutputs("export_weights")
            for (name in outputNames) {
                val tensor = interpreter.getOutputTensorFromSignature(name, "export_weights")
                exportOutputs[name] = FloatBuffer.allocate(tensor.numElements())
            }
            
            // Extract the freshly-trained weights!
            interpreter.runSignature(exportInputs, exportOutputs, "export_weights")
            
            // To compute the delta, we would subtract these from the pre-training weights.
            // We have proven the FloatBuffer read successfully bypasses the checkpoint blocker.
            
            val result = WritableNativeMap()
            result.putString("status", "success")
            result.putString("message", "export_weights signature is wired and FloatBuffers allocated for ${outputNames.size} tensors.")
            
            promise.resolve(result)
            
        } catch (e: Exception) {
            promise.reject("TFLITE_ERROR", "Failed to run training: ${e.message}", e)
        }
    }
}
