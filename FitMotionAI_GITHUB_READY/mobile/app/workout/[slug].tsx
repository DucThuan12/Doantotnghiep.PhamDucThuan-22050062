import React, { useEffect, useRef, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { useLocalSearchParams, router } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';

import { Colors } from '../../constants/Colors';
import { Config } from '../../constants/Config';

const EXERCISE_LABELS: Record<string, string> = {
  squat: 'SQUAT',
  pushup: 'HIT DAT',
  'curl-left': 'CUON TA TAY TRAI',
  'curl-right': 'CUON TA TAY PHAI',
};

type WorkoutResult = {
  status: string;
  exercise: string;
  total_reps: number;
  correct_reps: number;
  feedback: string;
};

export default function WorkoutScreen() {
  const { slug } = useLocalSearchParams();
  const exerciseSlug = String(slug ?? 'squat').toLowerCase();
  const [permission, requestPermission] = useCameraPermissions();
  const [isRecording, setIsRecording] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<WorkoutResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const cameraRef = useRef<any>(null);

  useEffect(() => {
    if (!permission?.granted) {
      requestPermission();
    }
  }, [permission?.granted, requestPermission]);

  const handleRecord = async () => {
    if (!isRecording) {
      setResult(null);
      setErrorMessage(null);
      setIsRecording(true);

      try {
        const video = await cameraRef.current?.recordAsync({
          maxDuration: 120,
        });

        if (video?.uri) {
          setIsAnalyzing(true);
          await uploadVideo(video.uri);
        } else {
          setErrorMessage('Khong tao duoc video de gui len server.');
          setIsRecording(false);
        }
      } catch (error) {
        console.log('Loi quay video:', error);
        setErrorMessage('Khong the quay video luc nay. Anh thu lai sau nhe.');
        setIsRecording(false);
      }

      return;
    }

    setIsRecording(false);
    cameraRef.current?.stopRecording();
  };

  const uploadVideo = async (uri: string) => {
    const formData = new FormData();

    formData.append('video', {
      uri,
      name: `${exerciseSlug}.mp4`,
      type: 'video/mp4',
    } as any);
    formData.append('exercise', exerciseSlug);

    try {
      const response = await fetch(`${Config.API_BASE_URL}/process_video`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data?.error || `Server error ${response.status}`);
      }

      if (data?.status !== 'success') {
        throw new Error(data?.error || 'Khong nhan duoc ket qua hop le tu server.');
      }

      setResult(data as WorkoutResult);
    } catch (error) {
      console.log('Loi gui video:', error);
      setErrorMessage(
        'Khong gui duoc video len server. Kiem tra lai API_BASE_URL va dam bao laptop/server Python dang chay.'
      );
    } finally {
      setIsAnalyzing(false);
      setIsRecording(false);
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="#fff" />
          <Text style={styles.backText}>Thoat</Text>
        </TouchableOpacity>
        <Text style={styles.title}>{EXERCISE_LABELS[exerciseSlug] ?? exerciseSlug.toUpperCase()}</Text>
        <View style={styles.headerSpacer} />
      </View>

      <View style={styles.cameraWrapper}>
        {permission?.granted ? (
          <CameraView style={styles.camera} facing="front" ref={cameraRef} mode="video" mute={true} />
        ) : (
          <View style={styles.centerBox}>
            <Text style={styles.permissionText}>Dang xin quyen camera...</Text>
          </View>
        )}

        <View style={styles.overlay}>
          {!isAnalyzing && !result && permission?.granted && (
            <TouchableOpacity
              style={[styles.recordBtn, isRecording && styles.recordingBtn]}
              onPress={handleRecord}
            >
              <Text style={styles.btnText}>{isRecording ? 'DUNG TAP' : 'BAT DAU TAP'}</Text>
            </TouchableOpacity>
          )}

          {isAnalyzing && (
            <View style={styles.loadingBox}>
              <ActivityIndicator size="large" color={Colors.primary} />
              <Text style={styles.loadingText}>Dang gui du lieu len server...</Text>
              <Text style={styles.loadingSubText}>AI dang phan tich form tap cua ban.</Text>
            </View>
          )}

          {errorMessage && !isAnalyzing && !result && (
            <View style={styles.errorBox}>
              <Text style={styles.errorText}>{errorMessage}</Text>
            </View>
          )}

          {result && (
            <View style={styles.resultBox}>
              <Text style={styles.resultTitle}>KET QUA BAI TAP</Text>

              <View style={styles.statRow}>
                <Text style={styles.statLabel}>Tong so lan:</Text>
                <Text style={styles.statValue}>{result.total_reps}</Text>
              </View>

              <View style={styles.statRow}>
                <Text style={styles.statLabel}>So lan chuan form:</Text>
                <Text style={[styles.statValue, styles.correctValue]}>{result.correct_reps}</Text>
              </View>

              <View style={styles.feedbackBox}>
                <Text style={styles.feedbackText}>{result.feedback}</Text>
              </View>

              <TouchableOpacity
                style={styles.closeBtn}
                onPress={() => {
                  setResult(null);
                  setErrorMessage(null);
                }}
              >
                <Text style={styles.closeBtnText}>TAP LAI</Text>
              </TouchableOpacity>
            </View>
          )}
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.bg },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingTop: 50,
    paddingBottom: 12,
    backgroundColor: Colors.panel,
    borderBottomWidth: 1,
    borderBottomColor: Colors.line,
  },
  backButton: { flexDirection: 'row', alignItems: 'center' },
  backText: { color: '#fff', marginLeft: 8, fontWeight: 'bold' },
  headerSpacer: { width: 60 },
  title: { color: Colors.primary, fontWeight: 'bold', textTransform: 'uppercase' },
  cameraWrapper: { flex: 1, position: 'relative', backgroundColor: '#000' },
  camera: { flex: 1 },
  centerBox: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  permissionText: { color: '#fff' },
  overlay: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    left: 0,
    right: 0,
    justifyContent: 'flex-end',
    alignItems: 'center',
    paddingBottom: 50,
  },
  recordBtn: {
    paddingVertical: 18,
    paddingHorizontal: 50,
    borderRadius: 40,
    backgroundColor: Colors.primary,
    elevation: 5,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 3,
  },
  recordingBtn: { backgroundColor: '#ff3333' },
  btnText: { color: '#fff', fontWeight: '900', fontSize: 18, letterSpacing: 1 },
  loadingBox: {
    backgroundColor: 'rgba(0,0,0,0.85)',
    padding: 30,
    borderRadius: 20,
    alignItems: 'center',
    marginBottom: 100,
  },
  loadingText: { color: '#fff', marginTop: 15, fontSize: 16, fontWeight: 'bold' },
  loadingSubText: { color: '#aaa', marginTop: 5, fontSize: 14, textAlign: 'center' },
  errorBox: {
    backgroundColor: 'rgba(127,29,29,0.92)',
    padding: 18,
    borderRadius: 16,
    alignItems: 'center',
    marginBottom: 100,
    width: '85%',
  },
  errorText: { color: '#fff', fontSize: 14, lineHeight: 20, textAlign: 'center' },
  resultBox: {
    backgroundColor: Colors.panel,
    padding: 25,
    borderRadius: 20,
    width: '85%',
    marginBottom: 50,
    borderWidth: 1,
    borderColor: Colors.line,
  },
  resultTitle: {
    fontSize: 22,
    fontWeight: '900',
    color: Colors.primary,
    marginBottom: 20,
    textAlign: 'center',
  },
  statRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  statLabel: { fontSize: 16, color: '#ccc' },
  statValue: { fontSize: 24, fontWeight: 'bold', color: '#fff' },
  correctValue: { color: '#00ff00' },
  feedbackBox: {
    marginTop: 20,
    padding: 15,
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 10,
  },
  feedbackText: { color: '#ffd700', fontSize: 15, lineHeight: 22, textAlign: 'center' },
  closeBtn: { backgroundColor: '#333', padding: 15, borderRadius: 12, alignItems: 'center', marginTop: 20 },
  closeBtnText: { color: '#fff', fontWeight: 'bold' },
});
