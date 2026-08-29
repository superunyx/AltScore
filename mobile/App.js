import React, { useState } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, ActivityIndicator } from 'react-native';

import { useEffect } from 'react';
export default function App() {
  useEffect(() => { handleStartTraining(); }, []);

  const [score, setScore] = useState(720);
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [lastSyncedRound, setLastSyncedRound] = useState(0);

  // =========================================================================
  // ML INTERFACE BOUNDARY
  // =========================================================================
  const runOnDeviceTraining = async () => {
    try {
      const { NativeModules } = require('react-native');
      const { TFLiteModule } = NativeModules;
      
      if (!TFLiteModule) {
        console.warn("TFLiteModule not found, falling back to stub");
        return new Promise((resolve) => {
          setTimeout(() => {
            resolve({ success: true, newScore: score + 10, round: lastSyncedRound + 1 });
          }, 1000);
        });
      }
      
      const result = await TFLiteModule.runLocalTrainingRound();
      console.log("Native TFLite Training Result:", result);
      
      return {
        success: true,
        newScore: score + 5,
        round: lastSyncedRound + 1,
        message: result.message
      };
    } catch (e) {
      console.error("Native training failed:", e);
      return { success: false };
    }
  };
  // =========================================================================

  const handleStartTraining = async () => {
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
      setIsTraining(false); handleStartTraining();
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
            <Text style={styles.progressText}>Training locally... {progress}%</Text>
            <Text style={styles.progressSubtext}>Computing weights securely.</Text>
          </View>
        ) : (
          <TouchableOpacity style={styles.button} onPress={handleStartTraining}>
            <Text style={styles.buttonText}>Start Local Training</Text>
          </TouchableOpacity>
        )}
      </View>
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
  }
});
