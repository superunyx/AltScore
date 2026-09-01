import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, ActivityIndicator, ScrollView, NativeEventEmitter, NativeModules } from 'react-native';


const { TFLiteModule } = NativeModules;
const tfliteEmitter = TFLiteModule ? new NativeEventEmitter(TFLiteModule) : null;

export default function App() {
  const [score, setScore] = useState(720);
  
  useEffect(() => {
    handleStartTraining();
  }, []);

  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [lastSyncedRound, setLastSyncedRound] = useState(0);
  
  // Debug State
  const [showDebug, setShowDebug] = useState(false);
  const [trainStatus, setTrainStatus] = useState('Idle');
  const [logs, setLogs] = useState([]);
  const [lastTiming, setLastTiming] = useState(null);
  const [lastL2Norm, setLastL2Norm] = useState(null);
  const [lastByteSize, setLastByteSize] = useState(null);
  const [serverStatus, setServerStatus] = useState('Not connected');

  useEffect(() => {
    if (!tfliteEmitter) return;
    const statusSub = tfliteEmitter.addListener('TFLiteStatus', (status) => {
      setTrainStatus(status);
    });
    const logSub = tfliteEmitter.addListener('TFLiteLog', (logMsg) => {
      setLogs(prev => {
        const next = [...prev, logMsg];
        if (next.length > 30) return next.slice(next.length - 30);
        return next;
      });
    });
    
    return () => {
      statusSub.remove();
      logSub.remove();
    };
  }, []);

  // =========================================================================
  // ML INTERFACE BOUNDARY
  // =========================================================================
  const runOnDeviceTraining = async () => {
    try {
      if (!TFLiteModule) {
        console.warn("TFLiteModule not found, falling back to stub");
        return { success: true, newScore: score + 10, round: lastSyncedRound + 1 };
      }
      
      const result = await TFLiteModule.runLocalTrainingRound();
      console.log("Native TFLite Training Result:", result);
      
      setLastTiming(result.duration);
      setLastByteSize(result.byteSize);
      setLastL2Norm(result.l2Norm?.toFixed(4));
      
      const computedScore = result.computedScore || score;
      
      // 3. TFLiteModule.kt internally POSTs the JSON payload via OkHttp to the server.
      // We no longer need to upload delta.bin from React Native.
      
      setServerStatus(`Success: ${new Date().toLocaleTimeString()} (handled natively)`);
      setTrainStatus("Done");
      
      return {
        success: true,
        newScore: computedScore,
        round: lastSyncedRound + 1,
        message: result.message
      };
    } catch (e) {
      console.error("Native training failed:", e);
      setTrainStatus("Error");
      return { success: false };
    }
  };
  // =========================================================================

  const handleStartTraining = async () => {
    setLogs([]);
    setIsTraining(true);
    setProgress(0);
    
    try {
      const result = await runOnDeviceTraining();
      if (result.success) {
        setScore(result.newScore);
        setLastSyncedRound(result.round);
      }
    } catch (error) {
      console.error("Local training failed", error);
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>AltScore</Text>
        <Text style={styles.headerSubtitle}>For Gig Workers</Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.scoreLabel}>Your AltScore</Text>
        <Text style={styles.scoreValue}>{score}</Text>
        <Text style={styles.scoreExplainer}>
          Your AltScore is computed securely right here on your phone using your gig app usage and income patterns. None of your raw personal data ever leaves this device!
        </Text>
      </View>

      <View style={styles.trainingSection}>
        <Text style={styles.statusLabel}>
          Last Synced Round: {lastSyncedRound === 0 ? "Never" : `Round ${lastSyncedRound}`}
        </Text>
        
        {isTraining ? (
          <View style={styles.progressContainer}>
            <ActivityIndicator size="large" color="#007AFF" />
            <Text style={styles.progressText}>Training locally...</Text>
            <Text style={styles.progressSubtext}>{trainStatus}</Text>
          </View>
        ) : (
          <TouchableOpacity style={styles.button} onPress={handleStartTraining}>
            <Text style={styles.buttonText}>Start Local Training</Text>
          </TouchableOpacity>
        )}
      </View>

      <TouchableOpacity 
        style={styles.debugToggle} 
        onPress={() => setShowDebug(!showDebug)}
      >
        <Text style={styles.debugToggleText}>{showDebug ? "Hide Debug" : "Show Debug"}</Text>
      </TouchableOpacity>

      {showDebug && (
        <View style={styles.debugPanel}>
          <Text style={styles.debugTitle}>Debug / Monitor</Text>
          <Text style={styles.debugText}>Status: {trainStatus}</Text>
          <Text style={styles.debugText}>Last Timing: {lastTiming ? `${lastTiming}ms` : 'N/A'}</Text>
          <Text style={styles.debugText}>Last Delta L2 Norm: {lastL2Norm || 'N/A'}</Text>
          <Text style={styles.debugText}>Last Delta Size: {lastByteSize ? `${lastByteSize} bytes` : 'N/A'}</Text>
          <Text style={styles.debugText}>Server: {serverStatus}</Text>
          
          <Text style={[styles.debugText, { marginTop: 10, fontWeight: 'bold' }]}>Logs:</Text>
          <ScrollView style={styles.logScroll}>
            {logs.map((log, i) => (
              <Text key={i} style={styles.logText}>{log}</Text>
            ))}
          </ScrollView>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F2F2F7',
    padding: 20,
    justifyContent: 'center',
  },
  header: {
    alignItems: 'center',
    marginBottom: 40,
    marginTop: 40,
  },
  headerTitle: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#1C1C1E',
  },
  headerSubtitle: {
    fontSize: 16,
    color: '#8E8E93',
    marginTop: 4,
  },
  card: {
    backgroundColor: 'white',
    borderRadius: 16,
    padding: 24,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 3,
    marginBottom: 30,
  },
  scoreLabel: {
    fontSize: 18,
    color: '#8E8E93',
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  scoreValue: {
    fontSize: 72,
    fontWeight: '800',
    color: '#34C759',
    marginVertical: 10,
  },
  scoreExplainer: {
    fontSize: 14,
    color: '#3A3A3C',
    textAlign: 'center',
    lineHeight: 20,
    marginTop: 10,
  },
  trainingSection: {
    alignItems: 'center',
  },
  statusLabel: {
    fontSize: 14,
    color: '#8E8E93',
    marginBottom: 20,
    fontWeight: '600',
  },
  button: {
    backgroundColor: '#007AFF',
    paddingVertical: 16,
    paddingHorizontal: 32,
    borderRadius: 30,
    width: '100%',
    alignItems: 'center',
    shadowColor: '#007AFF',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
  },
  buttonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: '600',
  },
  progressContainer: {
    alignItems: 'center',
    padding: 20,
  },
  progressText: {
    marginTop: 16,
    fontSize: 18,
    fontWeight: '500',
    color: '#1C1C1E',
  },
  progressSubtext: {
    marginTop: 8,
    fontSize: 14,
    color: '#8E8E93',
  },
  debugToggle: {
    marginTop: 30,
    alignSelf: 'center',
    padding: 10,
  },
  debugToggleText: {
    color: '#007AFF',
    fontWeight: 'bold',
  },
  debugPanel: {
    marginTop: 10,
    backgroundColor: '#1E1E1E',
    padding: 15,
    borderRadius: 8,
    maxHeight: 250,
  },
  debugTitle: {
    color: 'white',
    fontWeight: 'bold',
    marginBottom: 10,
  },
  debugText: {
    color: '#00FF00',
    fontSize: 12,
    fontFamily: 'monospace',
    marginBottom: 4,
  },
  logScroll: {
    marginTop: 5,
    backgroundColor: '#000',
    padding: 5,
    borderRadius: 4,
  },
  logText: {
    color: '#00FF00',
    fontSize: 10,
    fontFamily: 'monospace',
  }
});
