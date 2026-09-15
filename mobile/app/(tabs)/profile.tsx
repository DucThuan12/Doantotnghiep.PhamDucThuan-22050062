import React, { useState } from 'react';
import { StyleSheet, Text, View, ScrollView, TextInput, TouchableOpacity, Modal, FlatList } from 'react-native';
import { Colors } from '../../constants/Colors';
import { Ionicons } from '@expo/vector-icons';
import { useUser } from '../../context/UserContext'; // <-- GỌI CONTEXT VÀO ĐÂY

const HEALTH_OPTIONS = [
  "Không có vấn đề đặc biệt", "Thể trạng yếu", "Đau gối", "Đau vai", "Cổ tay yếu", "Đau lưng", "Huyết áp không ổn định"
];
const GOAL_OPTIONS = ["Tăng cơ", "Tập nhẹ", "Giảm mỡ"];

export default function ProfileScreen() {
  // THAY VÌ DÙNG useState, TA LẤY TỪ CONTEXT:
  const { health, setHealth, goal, setGoal } = useUser();
  
  // State quản lý Modal chọn chức năng
  const [isModalVisible, setModalVisible] = useState(false);
  const [modalType, setModalType] = useState<'health' | 'goal'>('health');

  const openPicker = (type: 'health' | 'goal') => {
    setModalType(type);
    setModalVisible(true);
  };

  const handleSelect = (val: string) => {
    if (modalType === 'health') setHealth(val);
    else setGoal(val);
    setModalVisible(false);
  };

  // Logic Khuyến nghị tự động dựa trên Thể Trạng
  let recommend = { fit: 'Tất cả bài tập', avoid: 'Không', warn: 'Hệ thống chưa ghi nhận rủi ro đáng kể.' };
  
  if (health === "Huyết áp không ổn định") {
    recommend = {
      fit: 'Cuốn tạ tay trái, Cuốn tạ tay phải, Squat',
      avoid: 'Hít đất',
      warn: 'Nên tránh bài tập cường độ cao liên tục khi huyết áp chưa ổn định.'
    };
  } else if (health === "Đau gối") {
    recommend = { fit: 'Hít đất, Cuốn tạ', avoid: 'Squat', warn: 'Hạn chế các bài gập gối chịu tải trọng lớn.' };
  } else if (health === "Cổ tay yếu" || health === "Đau vai") {
    recommend = { fit: 'Squat', avoid: 'Hít đất, Cuốn tạ nặng', warn: 'Cẩn thận với các bài chống tay chịu lực.' };
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      
      <View style={{ marginBottom: 20 }}>
        <Text style={styles.eyebrow}>Hồ sơ người tập</Text>
        <Text style={styles.title}>Thông tin cá nhân</Text>
      </View>

      {/* BOX THÔNG TIN CƠ BẢN */}
      <View style={styles.panel}>
        <View style={styles.row}>
          <View style={styles.halfCol}>
            <Text style={styles.label}>Tuổi</Text>
            <TextInput style={styles.input} defaultValue="21" keyboardType="numeric" />
          </View>
          <View style={styles.halfCol}>
            <Text style={styles.label}>Chiều cao (cm)</Text>
            <TextInput style={styles.input} defaultValue="160.0" keyboardType="numeric" />
          </View>
        </View>

        <View style={styles.row}>
          <View style={styles.halfCol}>
            <Text style={styles.label}>Cân nặng (kg)</Text>
            <TextInput style={styles.input} defaultValue="100.0" keyboardType="numeric" />
          </View>
          <View style={styles.halfCol}>
            <Text style={styles.label}>Mục tiêu</Text>
            <TouchableOpacity style={styles.pickerBtn} onPress={() => openPicker('goal')}>
              <Text style={styles.pickerText}>{goal}</Text>
              <Ionicons name="chevron-down" size={16} color={Colors.muted} />
            </TouchableOpacity>
          </View>
        </View>

        <Text style={styles.label}>Tình trạng sức khỏe</Text>
        <TouchableOpacity style={styles.pickerBtn} onPress={() => openPicker('health')}>
          <Text style={styles.pickerText}>{health}</Text>
          <Ionicons name="chevron-down" size={16} color={Colors.muted} />
        </TouchableOpacity>

        <TouchableOpacity style={styles.button}>
          <Text style={styles.buttonText}>Lưu hồ sơ</Text>
        </TouchableOpacity>
      </View>

      {/* BOX KHUYẾN NGHỊ AI */}
      <View style={[styles.panel, { marginTop: 20 }]}>
        <Text style={styles.eyebrow}>Khuyến nghị</Text>
        <Text style={styles.subTitle}>Bài tập phù hợp</Text>

        <View style={styles.row}>
          <View style={[styles.suggestBox, { borderColor: Colors.accent }]}>
            <Text style={styles.suggestLabel}>Phù hợp</Text>
            <Text style={styles.suggestText}>{recommend.fit}</Text>
          </View>
          <View style={[styles.suggestBox, { borderColor: Colors.warning }]}>
            <Text style={styles.suggestLabel}>Hạn chế</Text>
            <Text style={styles.suggestText}>{recommend.avoid}</Text>
          </View>
        </View>

        <View style={styles.warningBox}>
          <Text style={styles.warningLabel}>Cảnh báo</Text>
          <Text style={styles.warningDesc}>{recommend.warn}</Text>
        </View>
      </View>

      {/* BOTTOM SHEET MODAL CHỌN OPTIONS */}
      <Modal visible={isModalVisible} transparent animationType="slide">
        <TouchableOpacity style={styles.modalOverlay} activeOpacity={1} onPress={() => setModalVisible(false)}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>{modalType === 'health' ? 'Tình trạng sức khỏe' : 'Mục tiêu'}</Text>
            <FlatList
              data={modalType === 'health' ? HEALTH_OPTIONS : GOAL_OPTIONS}
              keyExtractor={(item) => item}
              renderItem={({ item }) => (
                <TouchableOpacity 
                  style={[styles.modalItem, item === (modalType === 'health' ? health : goal) && styles.modalItemActive]}
                  onPress={() => handleSelect(item)}
                >
                  <Text style={[styles.modalItemText, item === (modalType === 'health' ? health : goal) && styles.modalItemTextActive]}>{item}</Text>
                </TouchableOpacity>
              )}
            />
          </View>
        </TouchableOpacity>
      </Modal>

    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.bg },
  content: { padding: 20, paddingTop: 60, paddingBottom: 40 },
  eyebrow: { color: Colors.primary, textTransform: 'uppercase', fontWeight: 'bold', fontSize: 12, marginBottom: 4 },
  title: { fontSize: 26, color: Colors.text, fontWeight: 'bold' },
  subTitle: { fontSize: 18, color: Colors.text, fontWeight: 'bold', marginBottom: 16 },
  panel: { backgroundColor: Colors.panel, padding: 20, borderRadius: 16, borderWidth: 1, borderColor: Colors.line },
  row: { flexDirection: 'row', gap: 12, marginBottom: 16 },
  halfCol: { flex: 1 },
  label: { color: Colors.text, fontWeight: '600', marginBottom: 8, fontSize: 13 },
  input: { backgroundColor: Colors.bg, borderWidth: 1, borderColor: Colors.line, borderRadius: 8, padding: 12, color: Colors.white, fontSize: 14 },
  pickerBtn: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', backgroundColor: Colors.bg, borderWidth: 1, borderColor: Colors.line, borderRadius: 8, padding: 14, marginBottom: 16 },
  pickerText: { color: Colors.white, fontSize: 14 },
  button: { backgroundColor: Colors.primary, borderRadius: 8, padding: 14, alignItems: 'center', marginTop: 8 },
  buttonText: { color: Colors.white, fontWeight: 'bold', fontSize: 14 },
  suggestBox: { flex: 1, backgroundColor: Colors.bg, padding: 16, borderRadius: 8, borderWidth: 1, borderLeftWidth: 4 },
  suggestLabel: { color: Colors.muted, fontSize: 12, marginBottom: 4 },
  suggestText: { color: Colors.text, fontSize: 15, fontWeight: 'bold' },
  warningBox: { backgroundColor: Colors.bgSoft, padding: 16, borderRadius: 8, borderWidth: 1, borderColor: Colors.line, marginTop: 12 },
  warningLabel: { color: Colors.text, fontWeight: 'bold', marginBottom: 4 },
  warningDesc: { color: Colors.muted, fontSize: 13, lineHeight: 20 },
  
  modalOverlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.6)', justifyContent: 'flex-end' },
  modalContent: { backgroundColor: Colors.panel, borderTopLeftRadius: 20, borderTopRightRadius: 20, padding: 20, maxHeight: '60%' },
  modalTitle: { color: Colors.text, fontSize: 18, fontWeight: 'bold', marginBottom: 16 },
  modalItem: { paddingVertical: 16, borderBottomWidth: 1, borderBottomColor: Colors.line },
  modalItemActive: { backgroundColor: Colors.bg },
  modalItemText: { color: Colors.text, fontSize: 16 },
  modalItemTextActive: { color: Colors.primary, fontWeight: 'bold' }
});