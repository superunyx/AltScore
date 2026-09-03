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
import java.util.Date
import java.text.SimpleDateFormat
import java.util.Locale
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import org.json.JSONArray

import java.security.KeyFactory
import java.security.PublicKey
import java.security.spec.X509EncodedKeySpec
import java.security.spec.MGF1ParameterSpec
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.OAEPParameterSpec
import javax.crypto.spec.PSource
import android.util.Base64
import java.security.SecureRandom

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

                // 3. Prepare dummy data matching expected shape [1, 8] and [1, 1]
                val numWindows = 1
                val features = 8
                
                val xBuffer = FloatBuffer.allocate(numWindows * features)
                val dummySmsLogs = emptyList<Map<String, Any>>()
                val dummyAppUsage = emptyMap<String, Map<String, Any>>()
                val computedFeatures = computeRatioFeatures(dummySmsLogs, dummyAppUsage, 0L, 30L * 24 * 60 * 60 * 1000L)
                for (i in 0 until numWindows) {
                    xBuffer.put(computedFeatures)
                }
                
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
                val postNoiseL2Norm = sqrt(postL2.toDouble()).toFloat()
                
                emitLog("DP Stats: orig_l2=$originalL2Norm, clip=$clipFactor, post_l2=$postNoiseL2Norm")

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
                emitStatus("Fetching Public Key")
                val pubReq = Request.Builder().url("http://127.0.0.1:8000/public_key").get().build()
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
                emitStatus("Encrypting Payload")
                val innerPayload = JSONObject()
                innerPayload.put("weight_delta", weightDeltaJson)
                innerPayload.put("data_samples", numWindows)
                
                val keyGen = KeyGenerator.getInstance("AES")
                keyGen.init(256)
                val aesKey = keyGen.generateKey()
                
                val nonce = ByteArray(12)
                secureRandom.nextBytes(nonce)
                
                val cipher = Cipher.getInstance("AES/GCM/NoPadding")
                val gcmSpec = GCMParameterSpec(128, nonce)
                cipher.init(Cipher.ENCRYPT_MODE, aesKey, gcmSpec)
                
                val ciphertext = cipher.doFinal(innerPayload.toString().toByteArray(Charsets.UTF_8))
                
                val rsaCipher = Cipher.getInstance("RSA/ECB/OAEPPadding")
                val oaepParams = OAEPParameterSpec("SHA-256", "MGF1", MGF1ParameterSpec.SHA256, PSource.PSpecified.DEFAULT)
                rsaCipher.init(Cipher.ENCRYPT_MODE, rsaPub, oaepParams)
                val encryptedKey = rsaCipher.doFinal(aesKey.encoded)
                
                // 8. Send OkHttp POST
                emitStatus("POSTing JSON")
                val payloadJson = JSONObject()
                payloadJson.put("client_id", "device_001")
                payloadJson.put("encrypted_key", Base64.encodeToString(encryptedKey, Base64.NO_WRAP))
                payloadJson.put("nonce", Base64.encodeToString(nonce, Base64.NO_WRAP))
                payloadJson.put("ciphertext", Base64.encodeToString(ciphertext, Base64.NO_WRAP))

                val jsonStr = payloadJson.toString()
                val l2Norm = postNoiseL2Norm.toDouble()

                val body = jsonStr.toRequestBody("application/json; charset=utf-8".toMediaType())
                
                // Using 127.0.0.1 for Android emulator -> host
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

    private fun computeRatioFeatures(
        smsLogs: List<Map<String, Any>>, 
        appUsage: Map<String, Map<String, Any>>, 
        windowStart: Long, 
        windowEnd: Long
    ): FloatArray {
        val windowSms = smsLogs.filter { sms -> 
            val tsStr = sms["timestamp"] as String
            val format = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.US)
            val dt = format.parse(tsStr)!!.time
            dt >= windowStart && dt < windowEnd
        }.sortedBy { 
            val format = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.US)
            format.parse(it["timestamp"] as String)!!.time 
        }
    
        val incomes = mutableListOf<Float>()
        val expenses = mutableListOf<Float>()
        val incomeTimes = mutableListOf<Long>()
    
        for (sms in windowSms) {
            val amt = kotlin.math.abs((sms["amount"] as Double).toFloat())
            if (sms["type"] == "credit") {
                incomes.add(amt)
                val format = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.US)
                incomeTimes.add(format.parse(sms["timestamp"] as String)!!.time)
            } else if (sms["type"] == "debit") {
                expenses.add(amt)
            }
        }
    
        var IRI = 0.5f
        var ISI = 0.5f
        var EIR = 0.0f
        var SR = 0.0f
        var SF = 0.0f
        var TD = 0.0f
        var EC = 0.5f
        var lowConfidence = 0.0f
    
        val sumIncome = incomes.sum()
        val sumExpense = expenses.sum()
    
        if (incomes.size < 4) {
            lowConfidence = 1.0f
            val hours = mutableListOf<Float>()
            val numDays = ((windowEnd - windowStart) / (1000 * 60 * 60 * 24)).toInt()
            val dateFormat = SimpleDateFormat("yyyy-MM-dd", Locale.US)
            for (i in 0 until numDays) {
                val currDate = windowStart + i * 24 * 60 * 60 * 1000L
                val dateStr = dateFormat.format(Date(currDate))
                val usage = appUsage[dateStr]
                val h = (usage?.get("hours_active") as? Double)?.toFloat() ?: 0.0f
                hours.add(h)
            }
            val meanHours = if (hours.isNotEmpty()) hours.sum() / hours.size else 0.0f
            if (meanHours > 0) {
                val varHours = hours.map { (it - meanHours) * (it - meanHours) }.sum() / hours.size
                val stdHours = sqrt(varHours.toDouble()).toFloat()
                val cvHours = stdHours / meanHours
                EC = 1.0f / (1.0f + cvHours)
            } else {
                EC = 0.0f
            }
        } else {
            val gaps = mutableListOf<Float>()
            for (i in 1 until incomeTimes.size) {
                gaps.add((incomeTimes[i] - incomeTimes[i - 1]) / (1000.0f * 60.0f * 60.0f * 24.0f))
            }
            if (gaps.isNotEmpty()) {
                val meanGap = gaps.sum() / gaps.size
                if (meanGap > 0) {
                    val varGap = gaps.map { (it - meanGap) * (it - meanGap) }.sum() / gaps.size
                    val stdGap = sqrt(varGap.toDouble()).toFloat()
                    val cvGap = stdGap / meanGap
                    IRI = 1.0f / (1.0f + cvGap)
                }
            }
    
            val meanIncome = sumIncome / incomes.size
            if (meanIncome > 0) {
                val varIncome = incomes.map { (it - meanIncome) * (it - meanIncome) }.sum() / incomes.size
                val stdIncome = sqrt(varIncome.toDouble()).toFloat()
                val cvIncome = stdIncome / meanIncome
                ISI = 1.0f / (1.0f + cvIncome)
            }
    
            if (sumIncome > 0) {
                val eirRaw = sumExpense / sumIncome
                EIR = kotlin.math.max(0.0f, kotlin.math.min(2.0f, eirRaw))
            }
    
            if (sumIncome > 0) {
                SR = (sumIncome - sumExpense) / sumIncome
            }
    
            val midPoint = windowStart + 15 * 24 * 60 * 60 * 1000L
            var earlyIncomes = 0.0f
            var earlyExpenses = 0.0f
            var lateIncomes = 0.0f
            var lateExpenses = 0.0f
    
            for (sms in windowSms) {
                val amt = kotlin.math.abs((sms["amount"] as Double).toFloat())
                val format = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.US)
                val dt = format.parse(sms["timestamp"] as String)!!.time
                if (dt in windowStart until midPoint) {
                    if (sms["type"] == "credit") earlyIncomes += amt
                    else if (sms["type"] == "debit") earlyExpenses += amt
                } else if (dt in midPoint until windowEnd) {
                    if (sms["type"] == "credit") lateIncomes += amt
                    else if (sms["type"] == "debit") lateExpenses += amt
                }
            }
    
            if (earlyIncomes > 0 && lateIncomes > 0) {
                val srEarly = (earlyIncomes - earlyExpenses) / earlyIncomes
                val srLate = (lateIncomes - lateExpenses) / lateIncomes
                TD = srLate - srEarly
            }
        }
    
        var periodsWithTx = 0
        var shortfallPeriods = 0
        for (w in 0 until 5) {
            val pStart = windowStart + w * 7 * 24 * 60 * 60 * 1000L
            val pEndCand = windowStart + (w + 1) * 7 * 24 * 60 * 60 * 1000L
            val pEnd = kotlin.math.min(pEndCand, windowEnd)
            if (pStart >= pEnd) break
    
            var pInc = 0.0f
            var pExp = 0.0f
            var txCount = 0
            for (sms in windowSms) {
                val format = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.US)
                val dt = format.parse(sms["timestamp"] as String)!!.time
                if (dt in pStart until pEnd) {
                    txCount++
                    val amt = kotlin.math.abs((sms["amount"] as Double).toFloat())
                    if (sms["type"] == "credit") pInc += amt
                    else if (sms["type"] == "debit") pExp += amt
                }
            }
            if (txCount > 0) {
                periodsWithTx++
                if (pExp > pInc) {
                    shortfallPeriods++
                }
            }
        }
        if (periodsWithTx > 0) {
            SF = shortfallPeriods.toFloat() / periodsWithTx.toFloat()
        }
    
        return floatArrayOf(IRI, ISI, EIR, SR, SF, TD, EC, lowConfidence)
    }
}
