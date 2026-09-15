import React, { useState } from 'react';
import { StyleSheet, Text, View, TextInput, TouchableOpacity, KeyboardAvoidingView, Platform, ScrollView } from 'react-native';
import { router } from 'expo-router';
import { Colors } from '../../constants/Colors';

export default function RegisterScreen() {
  const [fullname, setFullname] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const handleRegister = () => {
    // Đăng ký xong tự động vào Dashboard
    router.replace('/(tabs)/dashboard');
  };

  return (
    <KeyboardAvoidingView style={styles.container} behavior={Platform.OS === 'ios' ? 'padding' : 'height'}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        
        <View style={styles.brandContainer}>
          <View style={styles.brandBadge}>
            <Text style={styles.brandText}>FitMotion AI</Text>
          </View>
          <Text style={styles.title}>Tạo tài khoản mới</Text>
          <Text style={styles.subtitle}>Khởi tạo hồ sơ luyện tập của bạn</Text>
        </View>

        <View style={styles.card}>
          <View style={styles.formGroup}>
            <Text style={styles.label}>Họ và tên</Text>
            <TextInput style={styles.input} placeholder="Nhập họ và tên" placeholderTextColor={Colors.muted} value={fullname} onChangeText={setFullname} />
          </View>

          <View style={styles.formGroup}>
            <Text style={styles.label}>Email</Text>
            <TextInput style={styles.input} placeholder="Nhập email" placeholderTextColor={Colors.muted} keyboardType="email-address" autoCapitalize="none" value={email} onChangeText={setEmail} />
          </View>

          <View style={styles.formGroup}>
            <Text style={styles.label}>Mật khẩu</Text>
            <TextInput style={styles.input} placeholder="Nhập mật khẩu" placeholderTextColor={Colors.muted} secureTextEntry value={password} onChangeText={setPassword} />
          </View>

          <TouchableOpacity style={styles.button} onPress={handleRegister} activeOpacity={0.8}>
            <Text style={styles.buttonText}>Đăng ký</Text>
          </TouchableOpacity>

          <View style={[styles.authLinks, { justifyContent: 'center', marginTop: 24 }]}>
            <TouchableOpacity onPress={() => router.push('/(auth)/login')}>
              <Text style={styles.linkText}>Đã có tài khoản? Đăng nhập</Text>
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