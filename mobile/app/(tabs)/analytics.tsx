import React from 'react';
import { StyleSheet, Text, View, ScrollView } from 'react-native';
import { Colors } from '../../constants/Colors';

export default function AnalyticsScreen() {
  // Dữ liệu giả lập cho 7 ngày qua
  const chartData = [
    { day: 'T2', val: 12 },
    { day: 'T3', val: 25 },
    { day: 'T4', val: 0 },
    { day: 'T5', val: 40 },
    { day: 'T6', val: 18 },
    { day: 'T7', val: 30 },
    { day: 'CN', val: 50 }, // Ngày có calo cao nhất
  ];
  
  const maxVal = Math.max(...chartData.map(d => d.val));

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.eyebrow}>Phân tích luyện tập</Text>
      <Text style={styles.title}>Thống kê chi tiết</Text>
      <Text style={styles.subtitle}>Tổng hợp lỗi kỹ thuật, tiến độ và lịch sử 7 ngày qua.</Text>

      <View style={styles.grid}>
        <View style={styles.statCard}>
          <Text style={styles.statLabel}>Tổng lỗi tuần</Text>
          <Text style={styles.statNumber}>12</Text>
        </View>
        <View style={styles.statCard}>
          <Text style={styles.statLabel}>Calo đã đốt</Text>
          <Text style={styles.statNumber}>340</Text>
        </View>
      </View>

      {/* Box Biểu đồ Cột */}
      <View style={styles.chartPanel}>
        <View style={{ marginBottom: 20 }}>
          <Text style={styles.panelTag}>TỔNG QUAN 7 NGÀY</Text>
          <Text style={styles.chartTitle}>Calo tiêu thụ theo tuần</Text>
        </View>
        
        <View style={styles.chartContainer}>
          {chartData.map((item, index) => {
            const barHeight = item.val === 0 ? 0 : (item.val / maxVal) * 100;
            return (
              <View key={index} style={styles.barCol}>
                <Text style={styles.barValText}>{item.val > 0 ? item.val : ''}</Text>
                <View style={styles.barBg}>
                  <View style={[styles.barFill, { height: `${barHeight}%`, backgroundColor: item.val === maxVal ? Colors.accent : Colors.primary }]} />
                </View>
                <Text style={styles.barLabel}>{item.day}</Text>
              </View>
            );
          })}
        </View>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.bg },
  content: { padding: 20, paddingTop: 60, paddingBottom: 40 },
  eyebrow: { color: Colors.primary, textTransform: 'uppercase', fontWeight: 'bold', fontSize: 12, marginBottom: 8 },
  title: { fontSize: 28, color: Colors.text, fontWeight: 'bold', marginBottom: 8 },
  subtitle: { fontSize: 14, color: Colors.muted, marginBottom: 24, lineHeight: 20 },
  grid: { flexDirection: 'row', gap: 16, marginBottom: 20 },
  statCard: { flex: 1, backgroundColor: Colors.panel, padding: 20, borderRadius: 12, borderWidth: 1, borderColor: Colors.line },
  statLabel: { color: Colors.muted, fontSize: 13, marginBottom: 8 },
  statNumber: { color: Colors.text, fontSize: 28, fontWeight: 'bold' },
  chartPanel: { backgroundColor: Colors.panel, padding: 20, borderRadius: 12, borderWidth: 1, borderColor: Colors.line },
  panelTag: { color: Colors.primary, fontSize: 11, fontWeight: 'bold', textTransform: 'uppercase', marginBottom: 4 },
  chartTitle: { color: Colors.text, fontSize: 18, fontWeight: 'bold' },
  chartContainer: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-end', height: 180, paddingTop: 20 },
  barCol: { alignItems: 'center', flex: 1 },
  barValText: { color: Colors.muted, fontSize: 10, marginBottom: 4, height: 14 },
  barBg: { height: 120, width: 24, backgroundColor: Colors.bg, borderRadius: 4, justifyContent: 'flex-end', overflow: 'hidden' },
  barFill: { width: '100%', borderRadius: 4 },
  barLabel: { color: Colors.text, fontSize: 12, marginTop: 8, fontWeight: '600' }
});