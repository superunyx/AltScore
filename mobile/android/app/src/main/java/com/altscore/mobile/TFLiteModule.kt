package com.altscore.mobile

import android.app.AppOpsManager
import android.content.Context
import android.content.Intent
import android.os.Process
import android.provider.Settings
import android.util.Log
import androidx.work.Constraints
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.NetworkType
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.WritableNativeMap
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.flex.FlexDelegate
import java.io.FileInputStream
import java.io.IOException
import java.nio.FloatBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.TimeUnit

class TFLiteModule(reactContext: ReactApplicationContext) : ReactContextBaseJavaModule(reactContext) {

    private val TAG = "TFLiteModule"
    private val client = OkHttpClient()

    override fun getName(): String {
        return "TFLiteModule"
    }

    @ReactMethod
    fun setServerBaseUrl(url: String) {
        val prefs = reactApplicationContext.getSharedPreferences("altscore_prefs", Context.MODE_PRIVATE)
        prefs.edit().putString("server_url", url).apply()
    }

    @ReactMethod
    fun setTrainingEnabled(enabled: Boolean) {
        val workManager = WorkManager.getInstance(reactApplicationContext)
        if (enabled) {
            val constraintsBuilder = Constraints.Builder()
                .setRequiredNetworkType(NetworkType.UNMETERED)
                .setRequiresCharging(true)

            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.M) {
                constraintsBuilder.setRequiresDeviceIdle(true)
            }

            val constraints = constraintsBuilder.build()
            
            val workRequest = PeriodicWorkRequestBuilder<TrainingWorker>(24, TimeUnit.HOURS)
                .setConstraints(constraints)
                .build()

            workManager.enqueueUniquePeriodicWork(
                "TrainingWorker",
                ExistingPeriodicWorkPolicy.REPLACE,
                workRequest
            )
            Log.d(TAG, "TrainingWorker scheduled")
        } else {
            workManager.cancelUniqueWork("TrainingWorker")
            Log.d(TAG, "TrainingWorker cancelled")
        }
    }

    @ReactMethod
    fun computeScoreOnly(promise: Promise) {
        Thread {
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

                // 2. Initialize Interpreter
                val options = Interpreter.Options()
                flexDelegate = FlexDelegate()
                options.addDelegate(flexDelegate)
                options.setNumThreads(1)
                interpreter = Interpreter(mappedByteBuffer, options)

                // 3. Prepare data
                val numWindows = 1
                val features = 8
                val xBuffer = FloatBuffer.allocate(numWindows * features)
                
                val windowEnd = System.currentTimeMillis()
                val windowStart = windowEnd - (30L * 24 * 60 * 60 * 1000L)
                
                val realSmsLogs = FeatureExtractor.getRealSmsLogs(reactApplicationContext, windowStart, windowEnd)
                val realAppUsage = FeatureExtractor.getRealAppUsage(reactApplicationContext, windowStart, windowEnd)
                val computedFeatures = FeatureExtractor.computeRatioFeatures(realSmsLogs, realAppUsage, windowStart, windowEnd)
                
                for (i in 0 until numWindows) {
                    xBuffer.put(computedFeatures)
                }

                val coldStart = computedFeatures[7] > 0.0f
                
                // 4. Run infer
                xBuffer.rewind()
                val inferInputs = mapOf<String, Any>("x" to xBuffer)
                val inferOutputs = mapOf<String, Any>("output" to FloatBuffer.allocate(1))
                interpreter.runSignature(inferInputs, inferOutputs, "infer")
                
                val outputBuffer = inferOutputs["output"] as FloatBuffer
                outputBuffer.rewind()
                val rawScore = outputBuffer.get()
                val computedScore = (rawScore * 1000).toInt()

                val result = WritableNativeMap()
                result.putString("status", "success")
                result.putInt("computedScore", computedScore)
                result.putBoolean("coldStart", coldStart)
                result.putDouble("l2Norm", rawScore.toDouble()) // just to maintain interface shape if needed
                
                promise.resolve(result)
            } catch (e: Exception) {
                promise.reject("ERR", e.message)
            } finally {
                interpreter?.close()
                flexDelegate?.close()
            }
        }.start()
    }

    @ReactMethod
    fun checkUsageStatsPermission(promise: Promise) {
        val appOps = reactApplicationContext.getSystemService(Context.APP_OPS_SERVICE) as AppOpsManager
        val mode = appOps.checkOpNoThrow(
            AppOpsManager.OPSTR_GET_USAGE_STATS,
            Process.myUid(),
            reactApplicationContext.packageName
        )
        promise.resolve(mode == AppOpsManager.MODE_ALLOWED)
    }

    @ReactMethod
    fun submitScoreForReview(score: Float, clientId: String, promise: Promise) {
        Thread {
            try {
                val prefs = reactApplicationContext.getSharedPreferences("altscore_prefs", Context.MODE_PRIVATE)
                val serverBaseUrl = prefs.getString("server_url", null) ?: throw IOException("Server not configured")
                
                // Fetch Public Key
                val pubReq = Request.Builder().url("${serverBaseUrl}/public_key").get().build()
                var pubPem = ""
                client.newCall(pubReq).execute().use { response ->
                    if (!response.isSuccessful) throw IOException("Failed to fetch public key: ${response.code}")
                    val obj = org.json.JSONObject(response.body?.string() ?: "")
                    pubPem = obj.getString("public_key_pem")
                }
                
                val pubPemClean = pubPem.replace("-----BEGIN PUBLIC KEY-----", "")
                                        .replace("-----END PUBLIC KEY-----", "").replace("\n", "").replace("\r", "")
                val pubBytes = android.util.Base64.decode(pubPemClean, android.util.Base64.DEFAULT)
                val spec = java.security.spec.X509EncodedKeySpec(pubBytes)
                val kf = java.security.KeyFactory.getInstance("RSA")
                val rsaPub = kf.generatePublic(spec)
                
                // Envelope Encryption
                val innerPayload = org.json.JSONObject()
                innerPayload.put("model_output", score.toDouble())
                innerPayload.put("consent_given", true)
                innerPayload.put("timestamp", System.currentTimeMillis())
                
                val keyGen = javax.crypto.KeyGenerator.getInstance("AES")
                keyGen.init(256)
                val aesKey = keyGen.generateKey()
                
                val nonce = ByteArray(12)
                java.security.SecureRandom().nextBytes(nonce)
                
                val cipher = javax.crypto.Cipher.getInstance("AES/GCM/NoPadding")
                val gcmSpec = javax.crypto.spec.GCMParameterSpec(128, nonce)
                cipher.init(javax.crypto.Cipher.ENCRYPT_MODE, aesKey, gcmSpec)
                
                val plaintext = innerPayload.toString().toByteArray(Charsets.UTF_8)
                val ciphertextBytes = cipher.doFinal(plaintext)
                
                val rsaCipher = javax.crypto.Cipher.getInstance("RSA/ECB/OAEPWithSHA-256AndMGF1Padding")
                val oaepSpec = javax.crypto.spec.OAEPParameterSpec("SHA-256", "MGF1", java.security.spec.MGF1ParameterSpec.SHA256, javax.crypto.spec.PSource.PSpecified.DEFAULT)
                rsaCipher.init(javax.crypto.Cipher.ENCRYPT_MODE, rsaPub, oaepSpec)
                val encryptedKeyBytes = rsaCipher.doFinal(aesKey.encoded)
                
                val envelope = org.json.JSONObject()
                envelope.put("client_id", clientId)
                envelope.put("encrypted_key", android.util.Base64.encodeToString(encryptedKeyBytes, android.util.Base64.NO_WRAP))
                envelope.put("nonce", android.util.Base64.encodeToString(nonce, android.util.Base64.NO_WRAP))
                envelope.put("ciphertext", android.util.Base64.encodeToString(ciphertextBytes, android.util.Base64.NO_WRAP))
                
                // POST to server
                val reqBody = envelope.toString().toRequestBody("application/json; charset=utf-8".toMediaType())
                val postReq = Request.Builder().url("${serverBaseUrl}/submit_score_for_review").post(reqBody).build()
                
                client.newCall(postReq).execute().use { response ->
                    if (!response.isSuccessful) {
                        promise.reject("ERR", "Server rejected score submission: ${response.code}")
                    } else {
                        promise.resolve(true)
                    }
                }
            } catch (e: Exception) {
                promise.reject("ERR", e.message)
            }
        }.start()
    }

    @ReactMethod
    fun openUsageStatsSettings(promise: Promise) {
        val intent = Intent(Settings.ACTION_USAGE_ACCESS_SETTINGS)
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        reactApplicationContext.startActivity(intent)
        promise.resolve(true)
    }

    @ReactMethod
    fun addListener(eventName: String) {}

    @ReactMethod
    fun removeListeners(count: Int) {}
}
