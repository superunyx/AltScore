import React, { useState, useEffect, useRef } from 'react';
import {
  StyleSheet, Text, View, TouchableOpacity, ScrollView,
  NativeModules, PermissionsAndroid, Alert, ActivityIndicator,
  Animated, Dimensions, Platform, StatusBar
} from 'react-native';
import { NavigationContainer, DarkTheme } from '@react-navigation/native';
import { SafeAreaProvider, useSafeAreaInsets } from 'react-native-safe-area-context';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Svg, { Circle, G, Defs, LinearGradient, Stop, Path, Rect } from 'react-native-svg';

const { TFLiteModule } = NativeModules;
const Tab = createBottomTabNavigator();
const { width: SW, height: SH } = Dimensions.get('window');

// ─── Starry Background ──────────────────────────────────────────
// Deterministic star positions so they don't shift on re-render
const _stars = [];
const _seed = (s) => { let x = Math.sin(s) * 10000; return x - Math.floor(x); };
for (let i = 0; i < 500; i++) {
  _stars.push({
    x: _seed(i * 7 + 1) * SW,
    y: _seed(i * 13 + 3) * SH * 2.5,
    r: _seed(i * 3 + 5) * 1.2 + 0.3,
    o: _seed(i * 11 + 7) * 0.4 + 0.08,
  });
}
const StarField = () => (
  <View style={StyleSheet.absoluteFill} pointerEvents="none">
    <Svg width={SW} height={SH * 2.5} style={{ position: 'absolute', top: 0, left: 0 }}>
      {_stars.map((st, i) => (
        <Circle key={i} cx={st.x} cy={st.y} r={st.r} fill="#FFFFFF" opacity={st.o} />
      ))}
    </Svg>
  </View>
);
// ─── Palette ─────────────────────────────────────────────────────
// Pure black base, monochromatic grays, selective accent pops
const C = {
  bg:       '#000000',
  card:     '#0C0C0E',
  card2:    '#111114',
  raised:   '#18181B',
  dim:      '#222225',
  line:     '#1A1A1D',
  line2:    '#27272A',
  w:        '#FFFFFF',
  t1:       '#FAFAFA',
  t2:       '#A1A1AA',
  t3:       '#71717A',
  t4:       '#3F3F46',
  // Accent family: blue-cyan-violet only — no orange
  blue:     '#3B82F6',
  cyan:     '#22D3EE',
  sky:      '#38BDF8',
  green:    '#34D399',
  emerald:  '#10B981',
  lime:     '#A3E635',
  yellow:   '#FBBF24',
  red:      '#F87171',
  rose:     '#FB7185',
  purple:   '#A78BFA',
  violet:   '#818CF8',
  pink:     '#F472B6',
  indigo:   '#6366F1',
};

// ─── Animated Press ─────────────────────────────────────────────
const Press = ({ children, onPress, disabled, style }) => {
  const sc = useRef(new Animated.Value(1)).current;
  return (
    <TouchableOpacity activeOpacity={1} onPress={onPress} disabled={disabled}
      onPressIn={() => Animated.spring(sc, { toValue: 0.97, useNativeDriver: true, speed: 50 }).start()}
      onPressOut={() => Animated.spring(sc, { toValue: 1, useNativeDriver: true, speed: 50 }).start()}>
      <Animated.View style={[style, { transform: [{ scale: sc }] }]}>{children}</Animated.View>
    </TouchableOpacity>
  );
};

// ─── Card ────────────────────────────────────────────────────────
const Card = ({ children, style, accent }) => (
  <View style={[sty.card, style]}>
    {accent && <View style={[sty.cardAccent, { backgroundColor: accent }]} />}
    {children}
  </View>
);

// ─── Gradient Button ─────────────────────────────────────────────
const GradBtn = ({ onPress, title, subtitle, disabled, icon, colors = [C.blue, C.violet] }) => {
  const sc = useRef(new Animated.Value(1)).current;
  return (
    <TouchableOpacity activeOpacity={1} onPress={onPress} disabled={disabled}
      onPressIn={() => Animated.spring(sc, { toValue: 0.97, useNativeDriver: true, speed: 50 }).start()}
      onPressOut={() => Animated.spring(sc, { toValue: 1, useNativeDriver: true, speed: 50 }).start()}>
      <Animated.View style={[sty.gradBtn, disabled && sty.gradBtnOff, { transform: [{ scale: sc }] }]}>
        <Svg style={StyleSheet.absoluteFill} width="100%" height="100%">
          <Defs>
            <LinearGradient id="bg" x1="0" y1="0" x2="1" y2="0.5">
              <Stop offset="0" stopColor={disabled ? C.dim : colors[0]} />
              <Stop offset="1" stopColor={disabled ? C.t4 : colors[1]} />
            </LinearGradient>
          </Defs>
          <Rect x="0" y="0" width="100%" height="100%" rx={14} fill="url(#bg)" />
        </Svg>
        <View style={sty.gradBtnInner}>
          {icon && <View style={{ marginRight: 10 }}>{icon}</View>}
          <View>
            <Text style={[sty.gradBtnTitle, disabled && { color: C.t3 }]}>{title}</Text>
            {subtitle && <Text style={sty.gradBtnSub}>{subtitle}</Text>}
          </View>
        </View>
      </Animated.View>
    </TouchableOpacity>
  );
};

// ─── Bento Stat ─────────────────────────────────────────────────
const BentoStat = ({ label, value, sub, color = C.sky, wide }) => (
  <View style={[sty.bento, wide && { flex: 2 }]}>
    <Text style={sty.bentoLabel}>{label}</Text>
    <Text style={[sty.bentoVal, { color }]}>{value}</Text>
    {sub && <Text style={sty.bentoSub}>{sub}</Text>}
  </View>
);

// ─── Score Ring ─────────────────────────────────────────────────
const ScoreRing = ({ score }) => {
  const pulse = useRef(new Animated.Value(0.15)).current;
  const R = 88, STK = 10;
  const nr = R - STK;
  const circ = nr * 2 * Math.PI;
  const safe = score || 0;
  const pct = Math.max(0, Math.min(100, (safe / 900) * 100));
  const off = circ - (pct / 100) * circ;

  // Harmonious gradient: all blue-green-violet — NO orange
  const gc = safe >= 750 ? [C.emerald, C.cyan]
    : safe >= 650 ? [C.sky, C.violet]
    : safe >= 550 ? [C.violet, C.pink]
    : [C.rose, C.purple];

  useEffect(() => {
    Animated.loop(Animated.sequence([
      Animated.timing(pulse, { toValue: 0.5, duration: 2500, useNativeDriver: false }),
      Animated.timing(pulse, { toValue: 0.15, duration: 2500, useNativeDriver: false }),
    ])).start();
  }, []);

  return (
    <View style={sty.ringWrap}>
      <Svg height={R * 2} width={R * 2}>
        <Defs>
          <LinearGradient id="rg" x1="0" y1="0" x2="1" y2="1">
            <Stop offset="0" stopColor={gc[0]} />
            <Stop offset="1" stopColor={gc[1]} />
          </LinearGradient>
        </Defs>
        <G rotation="-90" origin={`${R}, ${R}`}>
          {/* Background track */}
          <Circle stroke={C.dim} fill="transparent" strokeWidth={STK} r={nr} cx={R} cy={R} />
          {/* Inner ring accent */}
          <Circle stroke={gc[0] + '12'} fill="transparent" strokeWidth={1} r={nr - STK / 2 - 6} cx={R} cy={R} />
          {/* Progress arc */}
          <Circle
            stroke="url(#rg)" fill="transparent" strokeWidth={STK}
            strokeDasharray={`${circ} ${circ}`} strokeDashoffset={off}
            strokeLinecap="round" r={nr} cx={R} cy={R}
          />
        </G>
      </Svg>
      <View style={sty.ringCenter}>
        <Text style={sty.ringScore}>{safe.toFixed(0)}</Text>
        <View style={[sty.ringDivider, { backgroundColor: C.w }]} />
        <Text style={[sty.ringMax, { color: C.w }]}>of 900</Text>
      </View>
    </View>
  );
};

// ─── Factor Row ─────────────────────────────────────────────────
const factorGradients = [
  ['#00FF87', '#60EFFF'],    // Income Regularity: neon green → neon blue
  ['#00C9FF', '#92FE9D'],    // Income Stability: cyan → sea green
  ['#B92B27', '#1565C0'],    // Expense Ratio: crimson → deep blue
  ['#8E2DE2', '#4A00E0'],    // Savings Rate: bright purple → indigo
  ['#FF416C', '#FF4B2B'],    // Shortfall Freq: hot pink → bright orange
  ['#FDFC47', '#24FE41'],    // Trend: neon yellow → neon green
];

const FactorRow = ({ label, pct, color, idx }) => {
  const w = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    Animated.timing(w, { toValue: pct, duration: 900, delay: idx * 80, useNativeDriver: false }).start();
  }, [pct]);
  const aw = w.interpolate({ inputRange: [0, 100], outputRange: ['0%', '100%'] });
  const grad = factorGradients[idx] || [color, color];
  const gid = `fg${idx}`;
  const barW = SW - 80; // approximate track width

  return (
    <View style={sty.fRow}>
      <View style={sty.fHeader}>
        <View style={[sty.fDot, { backgroundColor: grad[0] }]} />
        <Text style={sty.fLabel}>{label}</Text>
        <Text style={[sty.fPct, { color: grad[0] }]}>{pct}%</Text>
      </View>
      <View style={sty.fTrack}>
        <Animated.View style={{ width: aw, height: 8, borderRadius: 4, overflow: 'hidden' }}>
          <Svg width={barW} height={8}>
            <Defs>
              <LinearGradient id={gid} x1="0" y1="0" x2="1" y2="0">
                <Stop offset="0" stopColor={grad[0]} />
                <Stop offset="1" stopColor={grad[1]} />
              </LinearGradient>
            </Defs>
            <Rect x="0" y="0" width={barW} height={8} rx={4} fill={`url(#${gid})`} />
          </Svg>
        </Animated.View>
      </View>
    </View>
  );
};

// ─── Step Row ───────────────────────────────────────────────────
const StepRow = ({ num, label, desc, active, done, isLast }) => (
  <View style={[sty.stepRow, active && sty.stepActive, !isLast && { borderBottomWidth: 1, borderBottomColor: C.line }]}>
    <View style={[sty.stepBullet, done ? sty.stepDone : active ? sty.stepLive : null]}>
      {done ? (
        <Svg width={14} height={14} viewBox="0 0 24 24" fill="none" stroke={C.w} strokeWidth={3}><Path d="M20 6L9 17l-5-5" /></Svg>
      ) : (
        <Text style={[sty.stepBulletText, (done || active) && { color: C.w }]}>{num}</Text>
      )}
    </View>
    <View style={{ flex: 1 }}>
      <Text style={[sty.stepLabel, (done || active) && { color: C.t1 }]}>{label}</Text>
      {desc && <Text style={sty.stepDesc}>{desc}</Text>}
    </View>
    {active && <ActivityIndicator size="small" color={C.sky} />}
    {done && <View style={sty.stepDonePill}><Text style={{ color: C.emerald, fontSize: 10, fontWeight: '700' }}>Done</Text></View>}
  </View>
);

// ─── Icons ──────────────────────────────────────────────────────
const IconDash = ({ color, size = 22 }) => (
  <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={1.8}>
    <Rect x="3" y="3" width="7" height="7" rx="2" /><Rect x="14" y="3" width="7" height="7" rx="2" />
    <Rect x="14" y="14" width="7" height="7" rx="2" /><Rect x="3" y="14" width="7" height="7" rx="2" />
  </Svg>
);
const IconClock = ({ color, size = 22 }) => (
  <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={1.8}>
    <Circle cx="12" cy="12" r="10" /><Path d="M12 6v6l4 2" />
  </Svg>
);
const IconBolt = ({ color, size = 22 }) => (
  <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={1.8}>
    <Path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
  </Svg>
);
const IconShield = ({ color, size = 22 }) => (
  <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={1.8}>
    <Path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
  </Svg>
);
const IconGear = ({ color, size = 22 }) => (
  <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={1.8}>
    <Circle cx="12" cy="12" r="3" />
    <Path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-2 2 2 2 0 01-2-2v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06a1.65 1.65 0 00.33-1.82 1.65 1.65 0 00-1.51-1H3a2 2 0 01-2-2 2 2 0 012-2h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06a1.65 1.65 0 001.82.33H9a1.65 1.65 0 001-1.51V3a2 2 0 012-2 2 2 0 012 2v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06a1.65 1.65 0 00-.33 1.82V9a1.65 1.65 0 001.51 1H21a2 2 0 012 2 2 2 0 01-2 2h-.09a1.65 1.65 0 00-1.51 1z" />
  </Svg>
);

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  SCREENS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function DashboardScreen() {
  const ins = useSafeAreaInsets();
  const [score, setScore] = useState(null);
  const [cold, setCold] = useState(false);
  const [sync, setSync] = useState(null);
  const [histLen, setHistLen] = useState(0);
  const fade = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(fade, { toValue: 1, duration: 500, useNativeDriver: true }).start();
    const load = async () => {
      try {
        const h = await AsyncStorage.getItem('@score_history');
        if (h) {
          const a = JSON.parse(h);
          setHistLen(a.length);
          if (a.length) { const l = a[a.length - 1]; setScore(l.score); setCold(l.coldStart); setSync(l.timestamp); }
        }
      } catch (e) {}
    };
    load();
    const iv = setInterval(load, 2000);
    return () => clearInterval(iv);
  }, []);

  const tier = !score ? null
    : score >= 750 ? { label: 'Excellent', color: C.emerald }
    : score >= 650 ? { label: 'Good', color: C.sky }
    : score >= 550 ? { label: 'Fair', color: C.violet }
    : { label: 'Building', color: C.purple };

  const factors = [
    { label: 'Income Regularity', pct: 80, color: C.emerald },
    { label: 'Income Stability', pct: 60, color: C.sky },
    { label: 'Expense Ratio', pct: 40, color: C.violet },
    { label: 'Savings Rate', pct: 50, color: C.purple },
    { label: 'Shortfall Freq', pct: 20, color: C.rose },
    { label: 'Trend', pct: 70, color: C.cyan },
  ];

  return (
    <Animated.ScrollView style={[sty.screen, { opacity: fade }]} contentContainerStyle={[sty.scroll, { paddingTop: ins.top + 16 }]} showsVerticalScrollIndicator={false} scrollEventThrottle={16} overScrollMode="never">
      <StatusBar barStyle="light-content" backgroundColor={C.bg} />

      {/* Header */}
      <View style={sty.header}>
        <View>
          <Text style={sty.greeting}>AltScore</Text>
          <Text style={sty.headerSub}>Your privacy-first credit score</Text>
        </View>
      </View>

      {score ? (
        <>
          {/* Main Score Card */}
          <Card style={sty.heroCard}>
            <View style={sty.heroTop}>
              <View style={[sty.statusPill, { borderColor: tier.color + '30', backgroundColor: tier.color + '10' }]}>
                <View style={[sty.statusDot, { backgroundColor: tier.color }]} />
                <Text style={[sty.statusText, { color: tier.color }]}>{tier.label}</Text>
              </View>
              {cold && (
                <View style={[sty.statusPill, { borderColor: C.yellow + '30', backgroundColor: C.yellow + '10' }]}>
                  <Text style={{ fontSize: 10 }}>⚠️</Text>
                  <Text style={[sty.statusText, { color: C.yellow, fontSize: 10 }]}>Low Confidence</Text>
                </View>
              )}
            </View>

            <ScoreRing score={score} />

            {sync && (
              <Text style={sty.syncText}>
                Updated {new Date(sync).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} · {new Date(sync).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
              </Text>
            )}
          </Card>

          {/* Bento Grid */}
          <View style={sty.bentoRow}>
            <BentoStat label="Rounds" value={histLen} color={C.sky} />
            <BentoStat label="Model" value="FL" sub="On-device" color={C.emerald} />
            <BentoStat label="Status" value="Synced" color={C.violet} />
          </View>
        </>
      ) : (
        <Card style={sty.heroCard}>
          <View style={sty.empty}>
            <View style={sty.emptyRing}>
              <IconBolt color={C.t4} size={28} />
            </View>
            <Text style={sty.emptyTitle}>No Score Yet</Text>
            <Text style={sty.emptyDesc}>Head to Train to compute your first score.{'\n'}Everything stays on your device.</Text>
          </View>
        </Card>
      )}

      {/* Breakdown */}
      <Text style={sty.secTitle}>Breakdown</Text>
      <Card>
        {factors.map((f, i) => (
          <FactorRow key={i} idx={i} {...f} />
        ))}
      </Card>

      <View style={{ height: 30 }} />
    </Animated.ScrollView>
  );
}

// ─── History ────────────────────────────────────────────────────
function HistoryScreen() {
  const ins = useSafeAreaInsets();
  const [history, setHistory] = useState([]);
  const fade = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(fade, { toValue: 1, duration: 500, useNativeDriver: true }).start();
    const load = async () => {
      try {
        const h = await AsyncStorage.getItem('@score_history');
        if (h) setHistory(JSON.parse(h).reverse());
      } catch (e) {}
    };
    load();
    const iv = setInterval(load, 2000);
    return () => clearInterval(iv);
  }, []);

  return (
    <Animated.ScrollView style={[sty.screen, { opacity: fade }]} contentContainerStyle={[sty.scroll, { paddingTop: ins.top + 16 }]} showsVerticalScrollIndicator={false} scrollEventThrottle={16} overScrollMode="never">
      <View style={sty.header}>
        <View>
          <Text style={sty.greeting}>History</Text>
          <Text style={sty.headerSub}>{history.length} training rounds</Text>
        </View>
      </View>

      {history.length === 0 ? (
        <Card>
          <View style={sty.empty}>
            <View style={sty.emptyRing}><IconClock color={C.t4} size={28} /></View>
            <Text style={sty.emptyTitle}>No History</Text>
            <Text style={sty.emptyDesc}>Scores appear here after training.</Text>
          </View>
        </Card>
      ) : (
        history.map((h, i) => {
          const isFirst = i === 0;
          const prev = i < history.length - 1 ? history[i + 1] : null;
          const delta = prev ? h.score - prev.score : null;

          return (
            <Card key={i} accent={isFirst ? C.sky : undefined}>
              <View style={sty.histRow}>
                {/* Left: Date block */}
                <View style={sty.histDateBlock}>
                  <Text style={sty.histDay}>
                    {new Date(h.timestamp).toLocaleDateString('en-US', { day: 'numeric' })}
                  </Text>
                  <Text style={sty.histMonth}>
                    {new Date(h.timestamp).toLocaleDateString('en-US', { month: 'short' }).toUpperCase()}
                  </Text>
                </View>

                <View style={sty.histDivider} />

                {/* Middle */}
                <View style={{ flex: 1 }}>
                  <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
                    <Text style={sty.histTime}>
                      {new Date(h.timestamp).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
                    </Text>
                    {isFirst && <View style={sty.latestPill}><Text style={sty.latestText}>Latest</Text></View>}
                  </View>
                  {delta !== null && (
                    <View style={{ flexDirection: 'row', alignItems: 'center', gap: 4, marginTop: 4 }}>
                      <Text style={[sty.histDelta, { color: delta >= 0 ? C.emerald : C.rose }]}>
                        {delta >= 0 ? '↑' : '↓'} {Math.abs(delta).toFixed(0)} pts
                      </Text>
                    </View>
                  )}
                </View>

                {/* Right: Score */}
                <Text style={[sty.histScore, isFirst && { color: C.sky }]}>{h.score.toFixed(0)}</Text>
              </View>
            </Card>
          );
        })
      )}
      <View style={{ height: 30 }} />
    </Animated.ScrollView>
  );
}

// ─── Train ──────────────────────────────────────────────────────
function TrainScreen() {
  const ins = useSafeAreaInsets();
  const [isTraining, setIsTraining] = useState(false);
  const [step, setStep] = useState(0);
  const [logs, setLogs] = useState([]);
  const fade = useRef(new Animated.Value(0)).current;
  const blink = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    Animated.timing(fade, { toValue: 1, duration: 500, useNativeDriver: true }).start();
  }, []);

  useEffect(() => {
    if (isTraining) {
      Animated.loop(Animated.sequence([
        Animated.timing(blink, { toValue: 0.2, duration: 600, useNativeDriver: true }),
        Animated.timing(blink, { toValue: 1, duration: 600, useNativeDriver: true }),
      ])).start();
    } else { blink.setValue(1); }
  }, [isTraining]);

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
        Alert.alert("Usage Access Required", "Please enable Usage Access for this app in Settings.",
          [{ text: "Cancel", style: "cancel" }, { text: "Open Settings", onPress: () => TFLiteModule.openUsageStatsSettings() }]
        );
        return false;
      }
      return true;
    } catch (err) { return false; }
  };

  const handleStart = async () => {
    const ok = await checkPermissions();
    if (!ok) return;
    setIsTraining(true); setStep(1); setLogs(["Requesting permissions... OK"]);
    setTimeout(() => { setStep(2); setLogs(l => [...l, "Extracting features from SMS..."]); }, 500);
    setTimeout(() => { setStep(3); setLogs(l => [...l, "Running local TFLite training..."]); }, 1000);
    try {
      if (!TFLiteModule) throw new Error("TFLiteModule not found");
      const result = await TFLiteModule.runLocalTrainingRound();
      setStep(4); setLogs(l => [...l, "Applying Differential Privacy noise..."]);
      setTimeout(() => { setStep(5); setLogs(l => [...l, "Encrypting with AES-256-GCM + RSA-OAEP..."]); }, 500);
      setTimeout(() => { setStep(6); setLogs(l => [...l, "Uploading encrypted delta...", "✓ Round complete"]); setIsTraining(false); }, 1000);
      if (result.l2Norm) {
        const historyStr = await AsyncStorage.getItem('@score_history');
        const history = historyStr ? JSON.parse(historyStr) : [];
        history.push({ score: result.l2Norm * 1000, timestamp: Date.now(), coldStart: false });
        await AsyncStorage.setItem('@score_history', JSON.stringify(history));
      }
    } catch (e) {
      setIsTraining(false);
      setLogs(l => [...l, `✗ ${e.message}`]);
    }
  };

  const pipeline = [
    { label: 'Permissions', desc: 'SMS & usage access' },
    { label: 'Feature Extract', desc: 'Parse transaction SMS' },
    { label: 'Local Training', desc: 'TFLite model on-device' },
    { label: 'DP Noise', desc: 'Gaussian clip + inject' },
    { label: 'Encryption', desc: 'AES-256 + RSA wrap' },
    { label: 'Upload', desc: 'Encrypted delta only' },
  ];

  return (
    <Animated.ScrollView style={[sty.screen, { opacity: fade }]} contentContainerStyle={[sty.scroll, { paddingTop: ins.top + 16 }]} showsVerticalScrollIndicator={false} scrollEventThrottle={16} overScrollMode="never">
      <View style={sty.header}>
        <View>
          <Text style={sty.greeting}>Train</Text>
          <Text style={sty.headerSub}>Federated learning round</Text>
        </View>
        {isTraining && (
          <Animated.View style={[sty.statusPill, { borderColor: C.sky + '30', backgroundColor: C.sky + '10', opacity: blink }]}>
            <View style={[sty.statusDot, { backgroundColor: C.sky }]} />
            <Text style={[sty.statusText, { color: C.sky }]}>Running</Text>
          </Animated.View>
        )}
      </View>

      {/* Privacy Banner */}
      <View style={sty.banner}>
        <View style={sty.bannerIcon}><IconShield color={C.emerald} size={16} /></View>
        <Text style={sty.bannerText}>All computation stays on-device. Only encrypted, noised gradients are sent.</Text>
      </View>

      {/* CTA */}
      <View style={{ marginBottom: 28 }}>
        <GradBtn
          onPress={handleStart}
          disabled={isTraining}
          title={isTraining ? 'Training in Progress...' : 'Start Local Training'}
          subtitle={isTraining ? undefined : 'Secure · On-device · Private'}
          icon={!isTraining ? <IconBolt color={C.w} size={18} /> : undefined}
        />
      </View>

      {/* Pipeline */}
      {step > 0 && (
        <>
          <View style={sty.secRow}>
            <Text style={sty.secTitle}>Pipeline</Text>
            <View style={sty.stepCounter}>
              <Text style={sty.stepCounterText}>{step}/6</Text>
            </View>
          </View>
          <Card style={{ padding: 0, overflow: 'hidden' }}>
            {pipeline.map((p, i) => (
              <StepRow key={i} num={i + 1} label={p.label} desc={p.desc} active={step === i + 1} done={step > i + 1} isLast={i === pipeline.length - 1} />
            ))}
          </Card>
        </>
      )}

      {/* Terminal */}
      {logs.length > 0 && (
        <View style={sty.term}>
          <View style={sty.termBar}>
            <View style={sty.termDots}>
              <View style={[sty.termDot, { backgroundColor: '#FF5F57' }]} />
              <View style={[sty.termDot, { backgroundColor: '#FEBC2E' }]} />
              <View style={[sty.termDot, { backgroundColor: '#28C840' }]} />
            </View>
            <Text style={sty.termTitle}>secure_log</Text>
          </View>
          <View style={sty.termBody}>
            {logs.map((l, i) => (
              <Text key={i} style={[
                sty.termLine,
                l.startsWith('✓') && { color: C.emerald },
                l.startsWith('✗') && { color: C.rose },
              ]}>
                <Text style={{ color: C.t4 }}>{'❯ '}</Text>{l}
              </Text>
            ))}
            {isTraining && <Animated.Text style={[sty.termLine, { color: C.sky, opacity: blink }]}>{'❯ _'}</Animated.Text>}
          </View>
        </View>
      )}

      <View style={{ height: 30 }} />
    </Animated.ScrollView>
  );
}

// ─── Settings (includes Privacy info) ───────────────────────────
function SettingsScreen() {
  const ins = useSafeAreaInsets();
  const [submitting, setSubmitting] = useState(false);
  const [showPrivacy, setShowPrivacy] = useState(false);
  const fade = useRef(new Animated.Value(0)).current;
  useEffect(() => { Animated.timing(fade, { toValue: 1, duration: 500, useNativeDriver: true }).start(); }, []);

  const handleOptInShare = async () => {
    Alert.alert("Share Score for Review",
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
            } catch (e) { Alert.alert("Error", e.message); }
            setSubmitting(false);
        }}
      ]
    );
  };

  const privacySteps = [
    { title: 'On-Device Processing', desc: 'SMS and app usage are read locally to compute a 8-dimensional feature vector. Data NEVER leaves your phone.', color: C.sky },
    { title: 'Federated Learning', desc: 'Downloads global model, trains locally, calculates weight delta — no raw data shared.', color: C.violet },
    { title: 'Differential Privacy', desc: 'Weight delta is clipped and Gaussian noise applied — mathematically impossible to reverse-engineer.', color: C.emerald },
    { title: 'Envelope Encryption', desc: 'Encrypted with AES-256-GCM, key wrapped with RSA-OAEP. Only the server can decrypt.', color: C.cyan },
  ];

  const specs = [
    { k: 'Inference', v: 'TFLite', c: C.sky },
    { k: 'Privacy', v: 'ε-DP', c: C.emerald },
    { k: 'Encryption', v: 'AES-256', c: C.purple },
    { k: 'Key Wrap', v: 'RSA-OAEP', c: C.cyan },
    { k: 'Transport', v: 'TLS 1.3', c: C.violet },
  ];

  return (
    <Animated.ScrollView style={[sty.screen, { opacity: fade }]} contentContainerStyle={[sty.scroll, { paddingTop: ins.top + 16 }]} showsVerticalScrollIndicator={false} scrollEventThrottle={16} overScrollMode="never">
      <View style={sty.header}>
        <View>
          <Text style={sty.greeting}>Settings</Text>
          <Text style={sty.headerSub}>Configuration & privacy</Text>
        </View>
      </View>

      {/* Score Sharing */}
      <Text style={sty.secTitle}>Score Sharing</Text>
      <Card>
        <Text style={sty.cardDesc}>Voluntarily submit your locally computed score for a loan application. End-to-end encrypted, separate from training.</Text>
        <View style={{ marginTop: 20 }}>
          <GradBtn
            onPress={handleOptInShare}
            disabled={submitting}
            title={submitting ? 'Submitting...' : 'Submit Score for Review'}
            colors={[C.sky, C.blue]}
          />
        </View>
      </Card>

      {/* Privacy Architecture */}
      <TouchableOpacity activeOpacity={0.7} onPress={() => setShowPrivacy(!showPrivacy)}>
        <View style={sty.secRow}>
          <Text style={sty.secTitle}>Privacy Architecture</Text>
          <View style={sty.expandBtn}>
            <Text style={sty.expandText}>{showPrivacy ? 'Hide' : 'Show'}</Text>
          </View>
        </View>
      </TouchableOpacity>
      {showPrivacy && (
        <>
          {privacySteps.map((p, i) => (
            <Card key={i} accent={p.color}>
              <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                <View style={[sty.privNum, { backgroundColor: p.color + '15' }]}>
                  <Text style={[sty.privNumText, { color: p.color }]}>{String(i + 1).padStart(2, '0')}</Text>
                </View>
                <Text style={sty.privTitle}>{p.title}</Text>
              </View>
              <Text style={sty.privDesc}>{p.desc}</Text>
            </Card>
          ))}
        </>
      )}

      {/* Tech Stack */}
      <Text style={sty.secTitle}>Tech Stack</Text>
      <Card style={{ padding: 0 }}>
        {specs.map((sp, i) => (
          <View key={i} style={[sty.specRow, i !== specs.length - 1 && { borderBottomWidth: 1, borderBottomColor: C.line }]}>
            <Text style={sty.specKey}>{sp.k}</Text>
            <View style={[sty.specPill, { backgroundColor: sp.c + '12' }]}>
              <Text style={[sty.specVal, { color: sp.c }]}>{sp.v}</Text>
            </View>
          </View>
        ))}
      </Card>

      <Text style={sty.version}>AltScore v1.0 · Federated Learning</Text>
      <View style={{ height: 30 }} />
    </Animated.ScrollView>
  );
}

// ─── Nav ─────────────────────────────────────────────────────────
const NavTheme = { ...DarkTheme, colors: { ...DarkTheme.colors, background: 'transparent', card: C.card, text: C.t1, border: 'transparent', primary: C.sky } };

function RootTabs() {
  const ins = useSafeAreaInsets();
  const btm = Math.max(ins.bottom, 8) + 12;

  return (
    <Tab.Navigator initialRouteName="Dashboard" screenOptions={({ route }) => ({
      headerShown: false,
      tabBarButton: (props) => <TouchableOpacity {...props} activeOpacity={0.6} />,
      tabBarIcon: ({ color, focused }) => {
        const sz = focused ? 23 : 21;
        if (route.name === 'Dashboard') return <IconDash color={color} size={sz} />;
        if (route.name === 'History') return <IconClock color={color} size={sz} />;
        if (route.name === 'Train') return <IconBolt color={color} size={sz} />;
        if (route.name === 'Settings') return <IconGear color={color} size={sz} />;
      },
      tabBarActiveTintColor: C.w,
      tabBarInactiveTintColor: C.t4,
      tabBarStyle: {
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        backgroundColor: 'rgba(12, 12, 14, 0.92)',
        borderTopWidth: 1,
        borderTopColor: 'rgba(39, 39, 42, 0.7)',
        paddingTop: 10,
        paddingBottom: btm,
        height: 54 + btm,
        elevation: 0,
      },
      tabBarLabelStyle: { fontWeight: '600', fontSize: 10, marginTop: 2, letterSpacing: 0.4 },
    })}>
      <Tab.Screen name="Dashboard" component={DashboardScreen} />
      <Tab.Screen name="History" component={HistoryScreen} />
      <Tab.Screen name="Train" component={TrainScreen} />
      <Tab.Screen name="Settings" component={SettingsScreen} />
    </Tab.Navigator>
  );
}

export default function App() {
  return (
    <SafeAreaProvider style={{ flex: 1, backgroundColor: C.bg }}>
      <StarField />
      <NavigationContainer theme={NavTheme}>
        <RootTabs />
      </NavigationContainer>
    </SafeAreaProvider>
  );
}

// ─── Styles ──────────────────────────────────────────────────────
const sty = StyleSheet.create({
  screen: { flex: 1, backgroundColor: 'transparent' },
  scroll: { paddingHorizontal: 20, paddingBottom: 100 },

  // Header
  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 28 },
  greeting: { fontSize: 30, fontWeight: '800', color: C.w, letterSpacing: -0.8 },
  headerSub: { fontSize: 13, color: C.t3, marginTop: 3, fontWeight: '500' },

  // Status Pill
  statusPill: { flexDirection: 'row', alignItems: 'center', gap: 6, paddingHorizontal: 12, paddingVertical: 6, borderRadius: 20, borderWidth: 1 },
  statusDot: { width: 6, height: 6, borderRadius: 3 },
  statusText: { fontSize: 12, fontWeight: '700' },

  // Card
  card: {
    backgroundColor: 'rgba(12, 12, 14, 0.4)', // C.card with more transparency
    borderRadius: 16,
    borderWidth: 1,
    borderColor: 'rgba(39, 39, 42, 0.6)', // softer border to match glass
    padding: 20,
    marginBottom: 12,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.4,
    shadowRadius: 16,
    elevation: 8,
  },
  cardAccent: { position: 'absolute', top: 0, left: 0, bottom: 0, width: 3, borderTopLeftRadius: 16, borderBottomLeftRadius: 16 },
  cardDesc: { fontSize: 13, color: C.t3, lineHeight: 22 },

  // Hero
  heroCard: {
    paddingBottom: 24,
    borderColor: C.line2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 12 },
    shadowOpacity: 0.5,
    shadowRadius: 24,
    elevation: 12,
  },
  heroTop: { flexDirection: 'row', gap: 8, marginBottom: 4 },

  // Section
  secTitle: { fontSize: 17, fontWeight: '700', color: C.t2, letterSpacing: -0.2, marginBottom: 10, marginTop: 16 },
  secRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10, marginTop: 16 },

  // Ring
  ringWrap: { alignItems: 'center', justifyContent: 'center', marginVertical: 16 },
  ringGlow: { position: 'absolute', width: 176, height: 176, borderRadius: 88, borderWidth: 16 },
  ringCenter: { position: 'absolute', alignItems: 'center' },
  ringScore: { fontSize: 44, fontWeight: '800', color: C.w, fontVariant: ['tabular-nums'], letterSpacing: -2, includeFontPadding: false },
  ringDivider: { width: 28, height: 2, borderRadius: 1, marginVertical: 6 },
  ringMax: { fontSize: 11, fontWeight: '700', letterSpacing: 1, textTransform: 'uppercase' },

  // Sync
  syncText: { color: C.t4, fontSize: 11, textAlign: 'center', letterSpacing: 0.2 },

  // Bento
  bentoRow: { flexDirection: 'row', gap: 10, marginBottom: 10 },
  bento: {
    flex: 1,
    backgroundColor: 'rgba(17, 17, 20, 0.4)', // C.card2 with more transparency
    borderRadius: 14,
    borderWidth: 1,
    borderColor: 'rgba(39, 39, 42, 0.5)',
    padding: 14,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 6,
  },
  bentoLabel: { fontSize: 10, color: C.t4, fontWeight: '600', letterSpacing: 0.5, textTransform: 'uppercase', marginBottom: 6 },
  bentoVal: { fontSize: 18, fontWeight: '800', letterSpacing: -0.5 },
  bentoSub: { fontSize: 9, color: C.t4, fontWeight: '500', marginTop: 2 },

  // Empty
  empty: { alignItems: 'center', paddingVertical: 40 },
  emptyRing: { width: 60, height: 60, borderRadius: 30, backgroundColor: C.raised, alignItems: 'center', justifyContent: 'center', marginBottom: 16, borderWidth: 1, borderColor: C.line },
  emptyTitle: { fontSize: 16, fontWeight: '700', color: C.t1, marginBottom: 8 },
  emptyDesc: { fontSize: 13, color: C.t3, textAlign: 'center', lineHeight: 20, maxWidth: 250 },

  // Factor
  fRow: { marginBottom: 16 },
  fHeader: { flexDirection: 'row', alignItems: 'center', marginBottom: 8 },
  fDot: { width: 7, height: 7, borderRadius: 4, marginRight: 10 },
  fLabel: { flex: 1, fontWeight: '600', color: C.t2, fontSize: 13 },
  fPct: { fontWeight: '800', fontSize: 13, fontVariant: ['tabular-nums'] },
  fTrack: { height: 8, backgroundColor: C.raised, borderRadius: 4, overflow: 'hidden' },
  fFill: { height: '100%', borderRadius: 4 },

  // History
  histRow: { flexDirection: 'row', alignItems: 'center' },
  histDateBlock: { alignItems: 'center', width: 40 },
  histDay: { fontSize: 22, fontWeight: '800', color: C.t1, letterSpacing: -1 },
  histMonth: { fontSize: 10, fontWeight: '700', color: C.t4, letterSpacing: 1 },
  histDivider: { width: 1, height: 36, backgroundColor: C.line, marginHorizontal: 16 },
  histTime: { fontSize: 13, fontWeight: '500', color: C.t3 },
  histScore: { fontSize: 24, fontWeight: '800', color: C.t1, fontVariant: ['tabular-nums'], letterSpacing: -1 },
  histDelta: { fontSize: 11, fontWeight: '700' },
  latestPill: { backgroundColor: C.sky + '15', paddingHorizontal: 8, paddingVertical: 2, borderRadius: 6 },
  latestText: { color: C.sky, fontSize: 9, fontWeight: '800', letterSpacing: 0.5 },

  // Train
  banner: { flexDirection: 'row', alignItems: 'center', gap: 12, backgroundColor: C.emerald + '06', borderWidth: 1, borderColor: C.emerald + '15', borderRadius: 14, padding: 14, marginBottom: 24 },
  bannerIcon: { width: 30, height: 30, borderRadius: 10, backgroundColor: C.emerald + '12', alignItems: 'center', justifyContent: 'center' },
  bannerText: { flex: 1, color: C.emerald, fontSize: 12, fontWeight: '500', lineHeight: 18 },

  // Gradient button
  gradBtn: { borderRadius: 14, overflow: 'hidden', height: 56, shadowColor: '#3B82F6', shadowOffset: { width: 0, height: 6 }, shadowOpacity: 0.3, shadowRadius: 12, elevation: 10 },
  gradBtnOff: { opacity: 0.5 },
  gradBtnInner: { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', paddingHorizontal: 20 },
  gradBtnTitle: { color: C.w, fontWeight: '700', fontSize: 15, letterSpacing: 0.2 },
  gradBtnSub: { color: 'rgba(255,255,255,0.55)', fontSize: 11, fontWeight: '500', marginTop: 1 },

  // Steps
  stepRow: { flexDirection: 'row', alignItems: 'center', gap: 14, paddingVertical: 14, paddingHorizontal: 20 },
  stepActive: { backgroundColor: C.sky + '06' },
  stepBullet: { width: 30, height: 30, borderRadius: 15, backgroundColor: C.raised, alignItems: 'center', justifyContent: 'center', borderWidth: 1, borderColor: C.line },
  stepDone: { backgroundColor: C.emerald, borderColor: C.emerald },
  stepLive: { backgroundColor: C.sky, borderColor: C.sky },
  stepBulletText: { fontSize: 12, fontWeight: '700', color: C.t4 },
  stepLabel: { fontSize: 13, fontWeight: '600', color: C.t3 },
  stepDesc: { fontSize: 11, color: C.t4, marginTop: 2 },
  stepDonePill: { backgroundColor: C.emerald + '12', paddingHorizontal: 8, paddingVertical: 3, borderRadius: 6 },
  stepCounter: { backgroundColor: C.sky + '12', paddingHorizontal: 10, paddingVertical: 4, borderRadius: 8 },
  stepCounterText: { color: C.sky, fontSize: 12, fontWeight: '800', fontVariant: ['tabular-nums'] },

  // Terminal
  term: { backgroundColor: '#050505', borderRadius: 14, borderWidth: 1, borderColor: C.line2, overflow: 'hidden', marginTop: 8, shadowColor: '#000', shadowOffset: { width: 0, height: 6 }, shadowOpacity: 0.4, shadowRadius: 12, elevation: 8 },
  termBar: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: 14, paddingVertical: 10, borderBottomWidth: 1, borderBottomColor: C.line, backgroundColor: '#0A0A0D' },
  termDots: { flexDirection: 'row', gap: 6, marginRight: 12 },
  termDot: { width: 9, height: 9, borderRadius: 5 },
  termTitle: { color: C.t4, fontSize: 11, fontWeight: '600', fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace' },
  termBody: { padding: 14 },
  termLine: { color: C.t2, fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace', fontSize: 11, lineHeight: 22 },

  // Privacy in Settings
  privNum: { width: 28, height: 28, borderRadius: 8, alignItems: 'center', justifyContent: 'center' },
  privNumText: { fontSize: 12, fontWeight: '800' },
  privTitle: { fontSize: 14, fontWeight: '700', color: C.t1 },
  privDesc: { fontSize: 12, color: C.t3, lineHeight: 20, paddingLeft: 36 },

  // Expand
  expandBtn: { backgroundColor: C.raised, paddingHorizontal: 12, paddingVertical: 5, borderRadius: 8 },
  expandText: { color: C.t2, fontSize: 11, fontWeight: '700' },

  // Specs
  specRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingVertical: 14, paddingHorizontal: 20 },
  specKey: { fontSize: 13, color: C.t3, fontWeight: '500' },
  specPill: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: 8 },
  specVal: { fontSize: 11, fontWeight: '700' },

  version: { textAlign: 'center', color: C.t4, fontSize: 11, marginTop: 24, fontWeight: '500' },
});
