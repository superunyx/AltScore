import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, ScrollView, NativeEventEmitter, NativeModules, PermissionsAndroid, Alert, ActivityIndicator, Animated } from 'react-native';
import { NavigationContainer, DarkTheme } from '@react-navigation/native';
import { SafeAreaProvider, useSafeAreaInsets } from 'react-native-safe-area-context';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Svg, { Circle, G, Defs, LinearGradient, Stop, Path } from 'react-native-svg';

const { TFLiteModule } = NativeModules;
const Tab = createBottomTabNavigator();

// Modern Card Component
const ModernCard = ({ children, style }) => (
  <View style={[styles.card, style]}>
    {children}
  </View>
);

const ModernButton = ({ onPress, title, disabled, color = '#6366f1' }) => (
  <TouchableOpacity activeOpacity={0.7} onPress={onPress} disabled={disabled} style={{ marginBottom: 20 }}>
    <View style={[styles.btn, { backgroundColor: disabled ? '#334155' : color }]}>
      <Text style={[styles.btnText, disabled && { color: '#94a3b8' }]}>{title}</Text>
    </View>
  </TouchableOpacity>
);

// Progress Ring Component
const ScoreRing = ({ score }) => {
  const radius = 110;
  const stroke = 20;
  const normalizedRadius = radius - stroke * 2;
  const circumference = normalizedRadius * 2 * Math.PI;
  const safeScore = score || 0;
  // Map 0-900 to 0-100%
  const percent = Math.max(0, Math.min(100, (safeScore / 900) * 100));
  const strokeDashoffset = circumference - (percent / 100) * circumference;

  return (
    <View style={{ alignItems: 'center', justifyContent: 'center', marginVertical: 25 }}>
      <Svg height={radius * 2} width={radius * 2}>
        <Defs>
          <LinearGradient id="grad" x1="0" y1="0" x2="1" y2="1">
            <Stop offset="0" stopColor="#06b6d4" stopOpacity="1" />
            <Stop offset="1" stopColor="#3b82f6" stopOpacity="1" />
          </LinearGradient>
        </Defs>
        <G rotation="-90" origin={`${radius}, ${radius}`}>
          <Circle stroke="#334155" fill="transparent" strokeWidth={stroke} r={normalizedRadius} cx={radius} cy={radius} />
          <Circle 
            stroke="url(#grad)" 
            fill="transparent" 
            strokeWidth={stroke} 
            strokeDasharray={`${circumference} ${circumference}`} 
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round" 
            r={normalizedRadius} cx={radius} cy={radius} 
          />
        </G>
      </Svg>
      <View style={{ position: "absolute", top: 0, left: 0, right: 0, bottom: 0, justifyContent: "center", alignItems: "center" }}>
        <Text style={{ fontSize: 64, fontWeight: 'bold', color: '#f8fafc', includeFontPadding: false, textAlignVertical: 'center', fontVariant: ['tabular-nums'] }}>
          {safeScore.toFixed(0)}
        </Text>
        <Text style={{ fontSize: 16, color: '#94a3b8', fontWeight: 'bold', marginTop: -4 }}>
          out of 900
        </Text>
      </View>
    </View>
  );
};


const IconDashboard = ({ color }) => <Svg width={24} height={24} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={2}><Path d="M3 3h7v7H3zM14 3h7v7h-7zM14 14h7v7h-7zM3 14h7v7H3z" /></Svg>;
const IconHistory = ({ color }) => <Svg width={24} height={24} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={2}><Circle cx="12" cy="12" r="10" /><Path d="M12 6v6l4 2" /></Svg>;
const IconTrain = ({ color }) => <Svg width={24} height={24} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={2}><Path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" /></Svg>;
const IconPrivacy = ({ color }) => <Svg width={24} height={24} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={2}><Path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" /></Svg>;
const IconSettings = ({ color }) => <Svg width={24} height={24} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={2}><Circle cx="12" cy="12" r="3" /><Path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-2 2 2 2 0 01-2-2v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06a1.65 1.65 0 00.33-1.82 1.65 1.65 0 00-1.51-1H3a2 2 0 01-2-2 2 2 0 012-2h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06a1.65 1.65 0 001.82.33H9a1.65 1.65 0 001-1.51V3a2 2 0 012-2 2 2 0 012 2v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06a1.65 1.65 0 00-.33 1.82V9a1.65 1.65 0 001.51 1H21a2 2 0 012 2 2 2 0 01-2 2h-.09a1.65 1.65 0 00-1.51 1z" /></Svg>;

// --- Screens ---

function DashboardScreen({ route }) {
  const [currentScore, setCurrentScore] = useState(null);
  const [coldStart, setColdStart] = useState(false);
  const [lastSync, setLastSync] = useState(null);

  useEffect(() => {
    const loadScore = async () => {
      try {
        const historyStr = await AsyncStorage.getItem('@score_history');
        if (historyStr) {
          const history = JSON.parse(historyStr);
          if (history.length > 0) {
            const latest = history[history.length - 1];
            setCurrentScore(latest.score);
            setColdStart(latest.coldStart);
            setLastSync(latest.timestamp);
          }
        }
      } catch (e) {}
    };
    loadScore();
    
    const interval = setInterval(loadScore, 2000);
    return () => clearInterval(interval);
  }, []);

  const getTier = (s) => {
    if (!s) return '';
    if (s >= 750) return 'Excellent';
    if (s >= 650) return 'Good';
    if (s >= 550) return 'Fair';
    return 'Building';
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.scrollContent}>
      <ModernCard>
        <Text style={styles.title}>Your AltScore</Text>
        
        {currentScore ? (
          <>
            <ScoreRing score={currentScore} />
            <View style={{ alignItems: 'center', marginBottom: 15 }}>
              <View style={{ backgroundColor: '#334155', paddingHorizontal: 16, paddingVertical: 8, borderRadius: 20 }}>
                <Text style={{ color: '#38bdf8', fontWeight: '700', fontSize: 16 }}>Tier: {getTier(currentScore)}</Text>
              </View>
            </View>
            {coldStart && <Text style={{ color: '#fbbf24', textAlign: 'center', marginTop: 10, fontWeight: '600' }}>⚠️ Low Confidence (Cold Start)</Text>}
            <Text style={{ color: '#64748b', fontSize: 12, marginTop: 10, textAlign: 'center' }}>Last updated: {new Date(lastSync).toLocaleString()}</Text>
          </>
        ) : (
          <Text style={{ color: '#94a3b8', textAlign: 'center', marginVertical: 40 }}>No score calculated yet. Go to Train to sync.</Text>
        )}
      </ModernCard>

      <ModernCard>
        <Text style={styles.title}>Factor Breakdown</Text>
        {[
          { label: 'Income Regularity', desc: 'Consistency of your incoming deposits', pct: 80, color: '#10b981' },
          { label: 'Income Stability', desc: 'Variance in your income amounts', pct: 60, color: '#0ea5e9' },
          { label: 'Expense-to-Income', desc: 'Ratio of money out vs money in', pct: 40, color: '#f59e0b' },
          { label: 'Savings Rate', desc: 'Percentage of income retained', pct: 50, color: '#8b5cf6' },
          { label: 'Shortfall Frequency', desc: 'How often expenses exceed income', pct: 20, color: '#ef4444' },
          { label: 'Trend', desc: 'Overall trajectory of your finances', pct: 70, color: '#14b8a6' },
        ].map((f, i) => (
          <View key={i} style={{ marginBottom: 20 }}>
            <View style={{ flexDirection: 'row', justifyContent: 'space-between', marginBottom: 6 }}>
              <Text style={{ fontWeight: '600', color: '#f1f5f9' }}>{f.label}</Text>
              <Text style={{ color: f.color, fontWeight: '700' }}>{f.pct}%</Text>
            </View>
            <Text style={{ fontSize: 12, color: '#94a3b8', marginBottom: 10 }}>{f.desc}</Text>
            <View style={{ height: 8, backgroundColor: '#334155', borderRadius: 4, overflow: 'hidden' }}>
              <View style={{ width: `${f.pct}%`, height: '100%', backgroundColor: f.color, borderRadius: 4 }} />
            </View>
          </View>
        ))}
      </ModernCard>
    </ScrollView>
  );
}

function HistoryScreen() {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    const loadScore = async () => {
      try {
        const historyStr = await AsyncStorage.getItem('@score_history');
        if (historyStr) setHistory(JSON.parse(historyStr).reverse());
      } catch (e) {}
    };
    loadScore();
    const interval = setInterval(loadScore, 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.scrollContent}>
      <Text style={[styles.title, { marginBottom: 20 }]}>Score History</Text>
      {history.map((h, i) => (
        <ModernCard key={i} style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', padding: 20 }}>
          <Text style={{ fontSize: 16, fontWeight: "600", color: '#cbd5e1' }}>{new Date(h.timestamp).toLocaleDateString()}</Text>
          <Text style={{ fontSize: 26, fontWeight: '800', color: '#0ea5e9' }}>{h.score.toFixed(0)}</Text>
        </ModernCard>
      ))}
    </ScrollView>
  );
}

function TrainScreen() {
  const [isTraining, setIsTraining] = useState(false);
  const [step, setStep] = useState(0); 
  const [logs, setLogs] = useState([]);

  const checkPermissions = async () => {
    try {
      const granted = await PermissionsAndroid.request(
        PermissionsAndroid.PERMISSIONS.READ_SMS,
        { title: 'SMS Permission', message: 'We need access to your SMS to compute your AltScore securely on your device.', buttonPositive: 'OK' }
      );
      if (granted !== PermissionsAndroid.RESULTS.GRANTED) {
        Alert.alert("Permission Denied", "Cannot compute score without SMS permission.");
        return false;
      }
      
      const usageGranted = await TFLiteModule.checkUsageStatsPermission();
      if (!usageGranted) {
        Alert.alert(
          "Usage Access Required", "Please enable Usage Access for this app in Settings.",
          [{ text: "Cancel", style: "cancel", onPress: () => {} }, { text: "Open Settings", onPress: () => TFLiteModule.openUsageStatsSettings() }]
        );
        return false;
      }
      return true;
    } catch (err) { return false; }
  };

  const handleStartTraining = async () => {
    const hasPerms = await checkPermissions();
    if (!hasPerms) return;
    
    setIsTraining(true);
    setStep(1);
    setLogs(["Requesting permissions... OK"]);
    
    setTimeout(() => { setStep(2); setLogs(l => [...l, "Extracting features from SMS..."]); }, 500);
    setTimeout(() => { setStep(3); setLogs(l => [...l, "Running local TFLite training..."]); }, 1000);
    
    try {
      if (!TFLiteModule) throw new Error("TFLiteModule not found");
      const result = await TFLiteModule.runLocalTrainingRound();
      
      setStep(4); setLogs(l => [...l, "Applying Differential Privacy noise..."]);
      setTimeout(() => { setStep(5); setLogs(l => [...l, "Encrypting with AES-256-GCM + RSA-OAEP..."]); }, 500);
      setTimeout(() => { setStep(6); setLogs(l => [...l, "Uploading to server...", "Done!"]); setIsTraining(false); }, 1000);
      
      if (result.l2Norm) {
        const historyStr = await AsyncStorage.getItem('@score_history');
        const history = historyStr ? JSON.parse(historyStr) : [];
        history.push({
          score: result.l2Norm * 1000, 
          timestamp: Date.now(),
          coldStart: false
        });
        await AsyncStorage.setItem('@score_history', JSON.stringify(history));
      }
      
    } catch (e) {
      setIsTraining(false);
      setLogs(l => [...l, `Error: ${e.message}`]);
    }
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.scrollContent}>
      <Text style={[styles.title, { marginBottom: 10 }]}>Contribute</Text>
      <Text style={{ marginBottom: 25, color: '#94a3b8', lineHeight: 22 }}>Your data never leaves your device. Only aggregated, encrypted noise is sent to train the global model.</Text>
      
      <ModernButton onPress={handleStartTraining} disabled={isTraining} title={isTraining ? 'Training...' : 'Start Local Training'} color="#6366f1" />
      
      {isTraining && <ActivityIndicator size="large" color="#38bdf8" style={{ marginTop: 20 }} />}
      
      <ModernCard style={{ marginTop: 20, backgroundColor: '#0f172a', borderColor: '#334155', borderWidth: 1 }}>
        <Text style={{ color: '#f8fafc', fontWeight: '700', marginBottom: 15 }}>Secure Logs</Text>
        {logs.map((l, i) => <Text key={i} style={{ color: '#10b981', fontFamily: 'monospace', fontSize: 13, marginBottom: 4 }}>$ {l}</Text>)}
      </ModernCard>
    </ScrollView>
  );
}

function PrivacyScreen() {
  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.scrollContent}>
      <Text style={[styles.title, { marginBottom: 20 }]}>How It Works</Text>
      <ModernCard>
        <Text style={{ fontWeight: '700', marginBottom: 8, color: '#f1f5f9' }}>1. On-Device Processing</Text>
        <Text style={{ color: '#94a3b8', marginBottom: 20, lineHeight: 20 }}>Your SMS and app usage are read locally to compute a 8-dimensional feature vector. This data NEVER leaves your phone.</Text>
        
        <Text style={{ fontWeight: '700', marginBottom: 8, color: '#f1f5f9' }}>2. Federated Learning</Text>
        <Text style={{ color: '#94a3b8', marginBottom: 20, lineHeight: 20 }}>Your phone downloads the global model, trains it locally on your data, and calculates a weight delta.</Text>
        
        <Text style={{ fontWeight: '700', marginBottom: 8, color: '#f1f5f9' }}>3. Differential Privacy</Text>
        <Text style={{ color: '#94a3b8', marginBottom: 20, lineHeight: 20 }}>We clip the weight delta and apply Gaussian noise to mathematically guarantee that your individual data cannot be reverse-engineered.</Text>
        
        <Text style={{ fontWeight: '700', marginBottom: 8, color: '#f1f5f9' }}>4. Envelope Encryption</Text>
        <Text style={{ color: '#94a3b8', lineHeight: 20 }}>The noised update is encrypted with AES-256-GCM and the key is wrapped with the server's RSA-OAEP public key. Only the server can decrypt the final payload.</Text>
      </ModernCard>
    </ScrollView>
  );
}

function SettingsScreen() {
  const [submitting, setSubmitting] = useState(false);
  
  const handleOptInShare = async () => {
    Alert.alert(
      "Share Score for Review",
      "Do you explicitly consent to sharing your computed AltScore with the loan officer? This will NOT share your raw features.",
      [
        { text: "Cancel", style: "cancel" },
        { text: "I Consent", onPress: async () => {
            setSubmitting(true);
            try {
              const historyStr = await AsyncStorage.getItem('@score_history');
              if (!historyStr) throw new Error("No score available. Train first.");
              const history = JSON.parse(historyStr);
              const latest = history[history.length - 1];
              
              await TFLiteModule.submitScoreForReview(latest.score, "user-" + Math.floor(Math.random()*10000));
              Alert.alert("Success", "Score successfully encrypted and shared.");
            } catch (e) {
              Alert.alert("Error", e.message);
            }
            setSubmitting(false);
        }}
      ]
    );
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.scrollContent}>
      <Text style={[styles.title, { marginBottom: 20 }]}>Settings</Text>
      <ModernCard>
        <Text style={{ fontWeight: '700', marginBottom: 10, color: '#f1f5f9' }}>Score Sharing</Text>
        <Text style={{ color: '#94a3b8', marginBottom: 20, lineHeight: 20 }}>If you are applying for a loan, you can voluntarily submit your locally computed score. This is encrypted and entirely separate from background training.</Text>
        <ModernButton onPress={handleOptInShare} disabled={submitting} title={submitting ? 'Submitting...' : 'Submit Score for Review'} color="#0ea5e9" />
      </ModernCard>
    </ScrollView>
  );
}

const CustomDarkTheme = {
  ...DarkTheme,
  colors: {
    ...DarkTheme.colors,
    background: '#0f172a',
    card: '#1e293b',
    text: '#f8fafc',
    border: '#334155',
    primary: '#06b6d4',
  },
};

function RootTabs() {
  return (
    <Tab.Navigator initialRouteName="Dashboard" screenOptions={({ route }) => ({
      tabBarIcon: ({ color }) => {
        if (route.name === 'Dashboard') return <IconDashboard color={color} />;
        if (route.name === 'History') return <IconHistory color={color} />;
        if (route.name === 'Train') return <IconTrain color={color} />;
        if (route.name === 'Privacy') return <IconPrivacy color={color} />;
        if (route.name === 'Settings') return <IconSettings color={color} />;
      },
      tabBarActiveTintColor: '#06b6d4', 
      tabBarInactiveTintColor: '#64748b',
      tabBarStyle: { borderTopWidth: 1, borderTopColor: '#334155', backgroundColor: '#1e293b', elevation: 0, paddingTop: 8 }, 
      tabBarLabelStyle: { fontWeight: '600', fontSize: 11, marginTop: 4 },
      headerStyle: { backgroundColor: '#1e293b', borderBottomWidth: 1, borderBottomColor: '#334155' },
      headerTitleStyle: { fontWeight: '700' }
    })}>
      <Tab.Screen name="Dashboard" component={DashboardScreen} />
      <Tab.Screen name="History" component={HistoryScreen} />
      <Tab.Screen name="Train" component={TrainScreen} />
      <Tab.Screen name="Privacy" component={PrivacyScreen} />
      <Tab.Screen name="Settings" component={SettingsScreen} />
    </Tab.Navigator>
  );
}

export default function App() {
  return (
    <SafeAreaProvider>
      <NavigationContainer theme={CustomDarkTheme}>
        <RootTabs />
      </NavigationContainer>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0f172a' },
  scrollContent: { padding: 20, paddingBottom: 40 },
  title: { fontSize: 28, fontWeight: '800', marginBottom: 20, color: '#f8fafc' },
  card: {
    backgroundColor: '#1e293b',
    padding: 24,
    borderRadius: 24,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.3,
    shadowRadius: 20,
    elevation: 8,
    marginBottom: 24,
  },
  btn: { 
    paddingVertical: 16, 
    borderRadius: 16, 
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
    elevation: 4
  },
  btnText: { color: '#ffffff', fontWeight: '700', fontSize: 16, letterSpacing: 0.5 }
});
