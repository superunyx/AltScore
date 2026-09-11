package com.altscore.mobile

import android.app.usage.UsageStatsManager
import android.content.Context
import android.provider.Telephony
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.regex.Pattern
import kotlin.math.sqrt

object FeatureExtractor {

    fun getRealAppUsage(context: Context, windowStart: Long, windowEnd: Long): Map<String, Map<String, Any>> {
        val usageStatsManager = context.getSystemService(Context.USAGE_STATS_SERVICE) as UsageStatsManager
        val stats = usageStatsManager.queryUsageStats(UsageStatsManager.INTERVAL_DAILY, windowStart, windowEnd)
        
        val appUsageMap = mutableMapOf<String, Map<String, Any>>()
        val dateFormat = SimpleDateFormat("yyyy-MM-dd", Locale.US)
        
        for (usage in stats) {
            val dateStr = dateFormat.format(Date(usage.firstTimeStamp))
            val activeHours = usage.totalTimeInForeground / (1000.0 * 60.0 * 60.0)
            
            val currentMap = appUsageMap[dateStr]?.toMutableMap() ?: mutableMapOf()
            val currentHours = (currentMap["hours_active"] as? Double) ?: 0.0
            currentMap["hours_active"] = currentHours + activeHours
            appUsageMap[dateStr] = currentMap
        }
        return appUsageMap
    }

    fun getRealSmsLogs(context: Context, windowStart: Long, windowEnd: Long): List<Map<String, Any>> {
        val smsLogs = mutableListOf<Map<String, Any>>()
        val cursor = context.contentResolver.query(
            Telephony.Sms.CONTENT_URI,
            arrayOf(Telephony.Sms.BODY, Telephony.Sms.DATE),
            "${Telephony.Sms.DATE} >= ? AND ${Telephony.Sms.DATE} <= ?",
            arrayOf(windowStart.toString(), windowEnd.toString()),
            Telephony.Sms.DEFAULT_SORT_ORDER
        )
        
        val debitPat1 = Pattern.compile("(?i)(?:debited|sent|paid|withdrawn|transferred\\s+from).*?(?:Rs\\.?|INR|₹)\\s?([\\d,]+\\.?\\d*)")
        val debitPat2 = Pattern.compile("(?i)(?:Rs\\.?|INR|₹)\\s?([\\d,]+\\.?\\d*)\\s*(?:is\\s+|was\\s+|has\\s+been\\s+)?(?:debited|sent|paid|withdrawn|transferred|Dr\\.?|DR\\b)")
        val creditPat1 = Pattern.compile("(?i)(?:credited|received|added).*?(?:Rs\\.?|INR|₹)\\s?([\\d,]+\\.?\\d*)")
        val creditPat2 = Pattern.compile("(?i)(?:Rs\\.?|INR|₹)\\s?([\\d,]+\\.?\\d*)\\s*(?:is\\s+|was\\s+|has\\s+been\\s+)?(?:credited|received|added|Cr\\.?|CR\\b)")
        
        val otpPat = Pattern.compile("(?i)otp|one time password")
        val promoPat = Pattern.compile("(?i)promo|offer|discount|cashback")
        val dateFormat = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.US)

        cursor?.use {
            val bodyIndex = it.getColumnIndexOrThrow(Telephony.Sms.BODY)
            val dateIndex = it.getColumnIndexOrThrow(Telephony.Sms.DATE)

            while (it.moveToNext()) {
                val body = it.getString(bodyIndex)
                val dateStr = dateFormat.format(Date(it.getLong(dateIndex)))
                
                if (otpPat.matcher(body).find() || promoPat.matcher(body).find()) {
                    continue
                } else {
                    var amountStr: String? = null
                    var isDebit = false
                    var isCredit = false
                    
                    val d1 = debitPat1.matcher(body)
                    val d2 = debitPat2.matcher(body)
                    val c1 = creditPat1.matcher(body)
                    val c2 = creditPat2.matcher(body)
                    
                    if (d1.find()) { amountStr = d1.group(1); isDebit = true }
                    else if (d2.find()) { amountStr = d2.group(1); isDebit = true }
                    else if (c1.find()) { amountStr = c1.group(1); isCredit = true }
                    else if (c2.find()) { amountStr = c2.group(1); isCredit = true }
                    
                    if ((isDebit || isCredit) && amountStr != null) {
                        try {
                            val amount = amountStr.replace(",", "").toDouble()
                            val typeStr = if (isDebit) "debit" else "credit"
                            smsLogs.add(mapOf("timestamp" to dateStr, "amount" to amount, "type" to typeStr))
                        } catch (e: Exception) {}
                    }
                }
            }
        }
        return smsLogs
    }

    fun computeRatioFeatures(
        smsLogs: List<Map<String, Any>>, 
        appUsage: Map<String, Map<String, Any>>, 
        windowStart: Long, 
        windowEnd: Long
    ): FloatArray {
        val windowSms = smsLogs.filter { sms -> 
            val tsStr = sms["timestamp"] as String
            val format = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.US)
            val dt = format.parse(tsStr)!!.time
            dt in windowStart until windowEnd
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
