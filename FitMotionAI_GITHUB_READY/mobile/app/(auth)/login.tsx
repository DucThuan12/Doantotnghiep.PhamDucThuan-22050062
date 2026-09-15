import React, { useState } from 'react';
import { StyleSheet, Text, View, TextInput, TouchableOpacity, KeyboardAvoidingView, Platform, ScrollView } from 'react-native';
import { router } from 'expo-router';
import { Colors } from '../../constants/Colors';

export default function LoginScreen() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const handleLogin = () => {
    // Chuyển hướng vào Dashboard sau khi đăng nhập thành công
    router.replace('/(tabs)/dashboard');
  };

  return (
    <KeyboardAvoidingView style={styles.container} behavior={Platform.OS === 'ios' ? 'padding' : 'height'}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        
        <View style={styles.brandContainer}>
          <View style={styles.brandBadge}>
            <Text style={styles.brandText}>FitMotion AI</Text>
          </View>
          <Text style={styles.title}>Đăng nhập hệ thống</Text>
          <Text style={styles.subtitle}>Tiếp tục với tài khoản của bạn</Text>
        </View>

        <View style={styles.card}>
          <View style={styles.formGroup}>
            <Text style={styles.label}>Email</Text>
            <TextInput 
              style={styles.input}
              placeholder="Nhập email"
              placeholderTextColor={Colors.muted}
              keyboardType="email-address"
              autoCapitalize="none"
              value={email}
              onChangeText={setEmail}
            />
          </View>

          <View style={styles.formGroup}>
            <Text style={styles.label}>Mật khẩu</Text>
            <TextInput 
              style={styles.input}
              placeholder="Nhập mật khẩu"
              placeholderTextColor={Colors.muted}
              secureTextEntry
              value={password}
              onChangeText={setPassword}
            />
          </View>

          <TouchableOpacity style={styles.button} onPress={handleLogin} activeOpacity={0.8}>
            <Text style={styles.buttonText}>Đăng nhập</Text>
          </TouchableOpacity>

          <View style={styles.authLinks}>
            <TouchableOpacity onPress={() => router.push('/(auth)/register')}>
              <Text style={styles.linkText}>Tạo tài khoản mới</Text>
            </TouchableOpacity>
            <TouchableOpacity onPress={() => router.push('/(auth)/forgot')}>
              <Text style={styles.linkText}>Quên mật khẩu</Text>
            </TouchableOpacity>
          </View>

          <View style={styles.demoBox}>
            <Text style={styles.demoText}><Text style={styles.boldText}>User demo:</Text> 22050062@...edu.vn / 123456</Text>
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
  subtitle: { fontSize: 16, color: Colors.muted, textAlign: 'center' },
  card: { backgroundColor: Colors.panel, borderRadius: 16, padding: 24, borderWidth: 1, borderColor: Colors.line },
  formGroup: { marginBottom: 16 },
  label: { color: Colors.text, fontWeight: '600', marginBottom: 8, fontSize: 14 },
  input: { backgroundColor: Colors.bg, borderWidth: 1, borderColor: Colors.line, borderRadius: 8, padding: 14, color: Colors.white, fontSize: 16 },
  button: { backgroundColor: Colors.primary, borderRadius: 8, padding: 16, alignItems: 'center', marginTop: 12 },
  buttonText: { color: Colors.white, fontWeight: 'bold', fontSize: 16 },
  authLinks: { flexDirection: 'row', justifyContent: 'space-between', marginTop: 20 },
  linkText: { color: Colors.primary, fontWeight: '500', fontSize: 14 },
  demoBox: { marginTop: 24, padding: 16, borderRadius: 8, backgroundColor: Colors.bgSoft, borderWidth: 1, borderColor: Colors.line },
  demoText: { color: Colors.muted, fontSize: 13, lineHeight: 22 },
  boldText: { fontWeight: 'bold' }
});