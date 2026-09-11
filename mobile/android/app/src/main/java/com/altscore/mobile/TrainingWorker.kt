package com.altscore.mobile

import android.content.Context
import android.util.Base64
import android.util.Log
import androidx.work.Worker
import androidx.work.WorkerParameters
import org.json.JSONArray
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.flex.FlexDelegate
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.nio.channels.FileChannel
import java.security.KeyFactory
import java.security.SecureRandom
import java.security.spec.MGF1ParameterSpec
import java.security.spec.X509EncodedKeySpec
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.OAEPParameterSpec
import javax.crypto.spec.PSource
import kotlin.math.sqrt
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody

class TrainingWorker(
    private val context: Context,
    workerParams: WorkerParameters
) : Worker(context, workerParams) {

    private val TAG = "TrainingWorker"
    private val client = OkHttpClient()

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

    override fun doWork(): Result {
        Log.d(TAG, "TrainingWorker started")

        val prefs = context.getSharedPreferences("altscore_prefs", Context.MODE_PRIVATE)
        val serverBaseUrl = prefs.getString("server_url", null)
        if (serverBaseUrl == null) {
            Log.e(TAG, "Server URL not configured")
            return Result.failure()
        }

        var interpreter: Interpreter? = null
        var flexDelegate: FlexDelegate? = null

        try {
            // 1. Load the model from assets
            val assetManager = context.assets
            val assetFileDescriptor = assetManager.openFd("base_model.tflite")
            
            val mappedByteBuffer = FileInputStream(assetFileDescriptor.fileDescriptor).use { fileInputStream ->
                val fileChannel = fileInputStream.channel
                fileChannel.map(FileChannel.MapMode.READ_ONLY, assetFileDescriptor.startOffset, assetFileDescriptor.declaredLength)
            }

            // 2. Initialize Interpreter
            val options = Interpreter.Options()
            flexDelegate = FlexDelegate()
            options.addDelegate(flexDelegate)
            options.setNumThreads(1)
            interpreter = Interpreter(mappedByteBuffer, options)

            // Export initial weights
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

            // 3. Prepare data
            val numWindows = 1
            val features = 8
            val xBuffer = FloatBuffer.allocate(numWindows * features)
            val windowEnd = System.currentTimeMillis()
            val windowStart = windowEnd - (30L * 24 * 60 * 60 * 1000L)
            
            val realSmsLogs = FeatureExtractor.getRealSmsLogs(context, windowStart, windowEnd)
            val realAppUsage = FeatureExtractor.getRealAppUsage(context, windowStart, windowEnd)
            val computedFeatures = FeatureExtractor.computeRatioFeatures(realSmsLogs, realAppUsage, windowStart, windowEnd)
            
            for (i in 0 until numWindows) {
                xBuffer.put(computedFeatures)
            }
            
            val yBuffer = FloatBuffer.allocate(numWindows * 1)
            for (i in 0 until numWindows) { yBuffer.put(0.5f) }
            
            val trainInputs: MutableMap<String, Any> = java.util.HashMap()
            val trainOutputs: MutableMap<String, Any> = java.util.HashMap()
            val lossBuffer = FloatBuffer.allocate(1)
            
            // 4. Train
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

            // 5. Compute delta
            val tunedOutputs: MutableMap<String, Any> = java.util.HashMap()
            for (name in outputNames) {
                val tensor = interpreter.getOutputTensorFromSignature(name, "export_weights")
                tunedOutputs[name] = FloatBuffer.allocate(tensor.numElements())
            }
            interpreter.runSignature(exportInputs, tunedOutputs, "export_weights")
            
            var flatDelta = FloatArray(0)
            var originalL2Norm = 0.0f
            val keys = outputNames.sorted()
            
            for (name in keys) {
                val buffer = tunedOutputs[name] as FloatBuffer
                buffer.rewind()
                val tunedArr = FloatArray(buffer.capacity())
                buffer.get(tunedArr)
                
                val initArr = initWeightsMap[name]!!
                for (i in tunedArr.indices) {
                    val diff = tunedArr[i] - initArr[i]
                    flatDelta += diff
                    originalL2Norm += diff * diff
                }
            }
            originalL2Norm = sqrt(originalL2Norm.toDouble()).toFloat()
            
            val clipNorm = 0.5f
            val noiseMultiplier = 0.05f
            val clipFactor = kotlin.math.min(1.0f, clipNorm / (originalL2Norm + 1e-12f))
            for (i in flatDelta.indices) {
                flatDelta[i] *= clipFactor
            }
            
            val noiseStd = noiseMultiplier * clipNorm
            val secureRandom = SecureRandom()
            var postL2 = 0.0f
            for (i in flatDelta.indices) {
                flatDelta[i] += (secureRandom.nextGaussian() * noiseStd).toFloat()
                postL2 += flatDelta[i] * flatDelta[i]
            }
            
            val weightDeltaJson = JSONObject()
            var offset = 0
            for (name in keys) {
                val tensor = interpreter.getOutputTensorFromSignature(name, "export_weights")
                val shape = tensor.shape()
                val size = tensor.numElements()
                
                val sliced = flatDelta.copyOfRange(offset, offset + size)
                offset += size
                
                val offsetArr = intArrayOf(0)
                val nestedArray = buildNestedJsonArray(sliced, shape, offsetArr, 0)
                weightDeltaJson.put(name, nestedArray)
            }

            // 6. Fetch Public Key
            val pubReq = Request.Builder().url("${serverBaseUrl}/public_key").get().build()
            var pubPem = ""
            client.newCall(pubReq).execute().use { response ->
                if (!response.isSuccessful) throw IOException("Failed to fetch public key: ${response.code}")
                val obj = JSONObject(response.body?.string() ?: "")
                pubPem = obj.getString("public_key_pem")
            }
            
            val pubPemClean = pubPem.replace("-----BEGIN PUBLIC KEY-----", "")
                                    .replace("-----END PUBLIC KEY-----", "").replace("\n", "")
            val pubBytes = Base64.decode(pubPemClean, Base64.DEFAULT)
            val spec = X509EncodedKeySpec(pubBytes)
            val kf = KeyFactory.getInstance("RSA")
            val rsaPub = kf.generatePublic(spec)
            
            // 7. Envelope Encryption
            val innerPayload = JSONObject()
            innerPayload.put("weight_delta", weightDeltaJson)
            innerPayload.put("data_samples", numWindows)
            
            val keyGen = KeyGenerator.getInstance("AES")
            keyGen.init(256)
            val aesKey = keyGen.generateKey()
            
            val nonce = ByteArray(12)
            SecureRandom().nextBytes(nonce)
            
            val cipher = Cipher.getInstance("AES/GCM/NoPadding")
            val gcmSpec = GCMParameterSpec(128, nonce)
            cipher.init(Cipher.ENCRYPT_MODE, aesKey, gcmSpec)
            
            val ciphertext = cipher.doFinal(innerPayload.toString().toByteArray(Charsets.UTF_8))
            
            val rsaCipher = Cipher.getInstance("RSA/ECB/OAEPPadding")
            val oaepParams = OAEPParameterSpec("SHA-256", "MGF1", MGF1ParameterSpec.SHA256, PSource.PSpecified.DEFAULT)
            rsaCipher.init(Cipher.ENCRYPT_MODE, rsaPub, oaepParams)
            val encryptedKey = rsaCipher.doFinal(aesKey.encoded)
            
            // 8. Send OkHttp POST
            val payloadJson = JSONObject()
            payloadJson.put("client_id", "device_001")
            payloadJson.put("encrypted_key", Base64.encodeToString(encryptedKey, Base64.NO_WRAP))
            payloadJson.put("nonce", Base64.encodeToString(nonce, Base64.NO_WRAP))
            payloadJson.put("ciphertext", Base64.encodeToString(ciphertext, Base64.NO_WRAP))

            val jsonStr = payloadJson.toString()
            val body = jsonStr.toRequestBody("application/json; charset=utf-8".toMediaType())
            
            val request = Request.Builder()
                .url("${serverBaseUrl}/submit_update")
                .post(body)
                .build()
            
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    throw IOException("Unexpected code $response: ${response.body?.string()}")
                }
            }

            Log.d(TAG, "TrainingWorker completed successfully")
            return Result.success()
            
        } catch (e: Throwable) {
            Log.e(TAG, "Error in TrainingWorker", e)
            return Result.retry()
        } finally {
            interpreter?.close()
            flexDelegate?.close()
        }
    }
}
