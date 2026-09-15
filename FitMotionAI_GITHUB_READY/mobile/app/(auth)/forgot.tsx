import React, { useState } from 'react';
import { StyleSheet, Text, View, TextInput, TouchableOpacity, KeyboardAvoidingView, Platform, ScrollView } from 'react-native';
import { router } from 'expo-router';
import { Colors } from '../../constants/Colors';

export default function ForgotScreen() {
  const [email, setEmail] = useState('');

  return (
    <KeyboardAvoidingView style={styles.container} behavior={Platform.OS === 'ios' ? 'padding' : 'height'}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        
        <View style={styles.brandContainer}>
          <View style={styles.brandBadge}>
            <Text style={styles.brandText}>FitMotion AI</Text>
          </View>
          <Text style={styles.title}>Khôi phục mật khẩu</Text>
          <Text style={styles.subtitle}>Nhập email để gửi yêu cầu đặt lại mật khẩu</Text>
        </View>

        <View style={styles.card}>
          <View style={styles.formGroup}>
            <Text style={styles.label}>Email</Text>
            <TextInput style={styles.input} placeholder="Nhập email" placeholderTextColor={Colors.muted} keyboardType="email-address" autoCapitalize="none" value={email} onChangeText={setEmail} />
          </View>

          <TouchableOpacity style={styles.button} activeOpacity={0.8}>
            <Text style={styles.buttonText}>Gửi yêu cầu</Text>
          </TouchableOpacity>

          <View style={[styles.authLinks, { justifyContent: 'center', marginTop: 24 }]}>
            <TouchableOpacity onPress={() => router.push('/(auth)/login')}>
              <Text style={styles.linkText}>Quay lại đăng nhập</Text>
            </TouchableOpacity>
          </View>
        </View>

      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.bg },
  scrollContent: { flexGrow: 1, justifyContent: 'center', padding: 24 },
  brandContainer: { alignItems: 'center', marginBottom: 32, marginTop: 40 },
  brandBadge: { backgroundColor: Colors.bgSoft, paddingVertical: 8, paddingHorizontal: 16, borderRadius: 8, borderWidth: 1, borderColor: Colors.line, marginBottom: 16 },
  brandText: { color: Colors.primary, fontWeight: '700', fontSize: 13, textTransform: 'uppercase' },
  title: { fontSize: 26, color: Colors.text, fontWeight: 'bold', marginBottom: 8, textAlign: 'center' },
  subtitle: { fontSize: 16, color: Colors.muted, textAlign: 'center', paddingHorizontal: 20 },
  card: { backgroundColor: Colors.panel, borderRadius: 16, padding: 24, borderWidth: 1, borderColor: Colors.line },
  formGroup: { marginBottom: 16 },
  label: { color: Colors.text, fontWeight: '600', marginBottom: 8, fontSize: 14 },
  input: { backgroundColor: Colors.bg, borderWidth: 1, borderColor: Colors.line, borderRadius: 8, padding: 14, color: Colors.white, fontSize: 16 },
  button: { backgroundColor: Colors.primary, borderRadius: 8, padding: 16, alignItems: 'center', marginTop: 12 },
  buttonText: { color: Colors.white, fontWeight: 'bold', fontSize: 16 },
  authLinks: { flexDirection: 'row', justifyContent: 'space-between', marginTop: 20 },
  linkText: { color: Colors.primary, fontWeight: '500', fontSize: 14 },
});