import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, ScrollView, TouchableOpacity } from 'react-native';
import { router } from 'expo-router';
import { Colors } from '../../constants/Colors';
import { Ionicons } from '@expo/vector-icons';
import { useUser } from '../../context/UserContext'; // <-- GỌI CONTEXT VÀO ĐÂY

export default function DashboardScreen() {
  // Lấy dữ liệu từ Context (Kho chứa dùng chung)
  const { health, goal } = useUser();
  
  const [weekDays, setWeekDays] = useState<any[]>([]);
  const [currentDate, setCurrentDate] = useState('');

  // Hàm tự động tạo lịch tuần hiện tại
  useEffect(() => {
    const curr = new Date();
    setCurrentDate(`${curr.getFullYear()}-${(curr.getMonth() + 1).toString().padStart(2, '0')}-${curr.getDate().toString().padStart(2, '0')}`);
    
    // Tìm ngày đầu tuần (Thứ 2)
    const firstDay = new Date(curr);
    const dayOfWeek = curr.getDay(); // CN là 0, T2 là 1...
    const diff = curr.getDate() - dayOfWeek + (dayOfWeek === 0 ? -6 : 1);
    firstDay.setDate(diff);

    const days = [];
    const shortDays = ['CN', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7'];
    
    for (let i = 0; i < 7; i++) {
      const next = new Date(firstDay);
      next.setDate(firstDay.getDate() + i);
      const dateStr = `${next.getDate().toString().padStart(2, '0')}/${(next.getMonth() + 1).toString().padStart(2, '0')}`;
      
      days.push({
        short: shortDays[next.getDay()],
        date: dateStr,
        count: 0,
        kcal: 0,
        active: next.getDate() === curr.getDate() // Highlight đúng ngày hôm nay
      });
    }
    setWeekDays(days);
  }, []);

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Hero Section */}
      <View style={styles.heroPanel}>
        <Text style={styles.eyebrow}>Hồ sơ luyện tập</Text>
        <Text style={styles.title}>Bảng điều khiển người dùng</Text>
        <Text style={styles.subtitle}>Theo dõi tiến độ, lịch tập, lỗi kỹ thuật và phân tích vận động trong một không gian trực quan hơn.</Text>
        
        <View style={styles.heroMetrics}>
          <View style={styles.heroMiniCard}>
            <Text style={styles.miniLabel}>Mục tiêu</Text>
            {/* THAY TEXT CỨNG BẰNG BIẾN ĐỘNG */}
            <Text style={styles.miniNumber}>{goal}</Text> 
          </View>
          <View style={styles.heroMiniCard}>
            <Text style={styles.miniLabel}>Thể trạng</Text>
            {/* THAY TEXT CỨNG BẰNG BIẾN ĐỘNG */}
            <Text style={styles.miniNumber}>{health}</Text> 
          </View>
        </View>
      </View>

      {/* Lịch tập trong tuần (Tự động) */}
      <View style={styles.panel}>
        <View style={styles.panelHead}>
          <View>
            <Text style={styles.panelTag}>Lịch luyện tập</Text>
            <Text style={styles.panelTitle}>Kế hoạch trong tuần</Text>
          </View>
          <TouchableOpacity style={styles.btnSmall} onPress={() => router.push('/(tabs)/exercises')}>
            <Text style={styles.btnSmallText}>Bắt đầu buổi tập</Text>
          </TouchableOpacity>
        </View>

        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.weekRow}>
          {weekDays.map((day, index) => (
            <TouchableOpacity key={index} style={[styles.dayCard, day.active && styles.dayCardActive]}>
              <Text style={[styles.dayDate, day.active && styles.textWhite]}>{day.date}</Text>
              <Text style={[styles.dayMetric, day.active && styles.textWhite]}>{day.count} bài</Text>
              <Text style={[styles.daySub, day.active && styles.textWhite]}>{day.kcal} kcal</Text>
            </TouchableOpacity>
          ))}
        </ScrollView>
        
        {/* Box chi tiết ngày */}
        <View style={styles.dayDetailBox}>
          <Text style={styles.miniLabel}>CHI TIẾT THEO NGÀY</Text>
          <Text style={styles.miniNumber}>{currentDate}</Text>
          <View style={styles.emptyWarning}>
            <Text style={styles.emptyText}>Chưa có bài tập nào được lên lịch cho ngày này.</Text>
          </View>
        </View>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.bg },
  content: { padding: 20, paddingTop: 60, paddingBottom: 40, gap: 20 },
  eyebrow: { color: Colors.primary, textTransform: 'uppercase', fontWeight: 'bold', fontSize: 12, marginBottom: 8 },
  title: { fontSize: 26, color: Colors.text, fontWeight: 'bold', marginBottom: 8 },
  subtitle: { fontSize: 14, color: Colors.muted, lineHeight: 20, marginBottom: 20 },
  heroPanel: { marginBottom: 8 },
  heroMetrics: { flexDirection: 'row', gap: 12 },
  heroMiniCard: { flex: 1, backgroundColor: Colors.panel, padding: 16, borderRadius: 8, borderWidth: 1, borderColor: Colors.line },
  miniLabel: { color: Colors.muted, fontSize: 12, marginBottom: 4, textTransform: 'uppercase' },
  miniNumber: { color: Colors.text, fontSize: 16, fontWeight: 'bold' },
  panel: { backgroundColor: Colors.panel, padding: 20, borderRadius: 12, borderWidth: 1, borderColor: Colors.line },
  panelHead: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 },
  panelTag: { color: Colors.primary, fontSize: 12, fontWeight: 'bold', textTransform: 'uppercase', marginBottom: 4 },
  panelTitle: { color: Colors.text, fontSize: 18, fontWeight: 'bold' },
  btnSmall: { backgroundColor: Colors.primary, paddingHorizontal: 16, paddingVertical: 8, borderRadius: 6 },
  btnSmallText: { color: Colors.white, fontWeight: 'bold', fontSize: 12 },
  weekRow: { gap: 8, paddingBottom: 8 },
  dayCard: { backgroundColor: Colors.bg, padding: 12, borderRadius: 8, borderWidth: 1, borderColor: Colors.line, alignItems: 'center', minWidth: 70 },
  dayCardActive: { backgroundColor: Colors.primary, borderColor: Colors.primary },
  dayDate: { fontSize: 14, fontWeight: 'bold', color: Colors.text, marginBottom: 4 },
  dayMetric: { fontSize: 13, fontWeight: '600', color: Colors.text },
  daySub: { fontSize: 11, color: Colors.muted, marginTop: 2 },
  textWhite: { color: Colors.white },
  dayDetailBox: { marginTop: 16, borderTopWidth: 1, borderTopColor: Colors.line, paddingTop: 16 },
  emptyWarning: { marginTop: 12, padding: 12, borderWidth: 1, borderColor: Colors.line, borderRadius: 8, backgroundColor: Colors.bg },
  emptyText: { color: Colors.muted, fontSize: 13 }
});