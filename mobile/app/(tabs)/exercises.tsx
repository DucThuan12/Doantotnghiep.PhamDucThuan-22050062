import React, { useState } from 'react';
import { StyleSheet, Text, View, FlatList, TextInput, TouchableOpacity } from 'react-native';
import { router } from 'expo-router';
import { Colors } from '../../constants/Colors';
import { Ionicons } from '@expo/vector-icons';

export default function ExercisesScreen() {
  const [search, setSearch] = useState('');

  const exercises = [
    { id: '1', slug: 'squat', name: 'Squat', diff: 'Cơ bản', muscle: 'Đùi trước, mông, bắp chân', age: '15 - 100', kcal: '0.1', desc: 'Bài tập thân dưới giúp phát triển sức mạnh chân và mông.' },
    { id: '2', slug: 'pushup', name: 'Hít đất', diff: 'Trung bình', muscle: 'Ngực, vai, tay sau', age: '16 - 60', kcal: '0.12', desc: 'Bài tập thân trên giúp phát triển ngực, vai và tay sau.' },
    { id: '3', slug: 'curl-left', name: 'Cuốn tạ tay trái', diff: 'Cơ bản', muscle: 'Tay trước', age: '15 - 100', kcal: '0.08', desc: 'Bài tập đơn tay giúp phát triển bắp tay trước bên trái.' },
    { id: '4', slug: 'curl-right', name: 'Cuốn tạ tay phải', diff: 'Cơ bản', muscle: 'Tay trước', age: '15 - 100', kcal: '0.08', desc: 'Bài tập đơn tay giúp phát triển bắp tay trước bên phải.' },
  ];

  // Logic Search Realtime: Lọc ngay khi người dùng gõ phím
  const filteredExercises = exercises.filter(ex => 
    ex.name.toLowerCase().includes(search.toLowerCase()) || 
    ex.muscle.toLowerCase().includes(search.toLowerCase())
  );

  const renderItem = ({ item }: any) => (
    <View style={styles.card}>
      <View style={styles.cardHeader}>
        <Text style={styles.cardTitle}>{item.name}</Text>
        <View style={styles.diffPill}>
          <Text style={styles.diffText}>{item.diff}</Text>
        </View>
      </View>
      <Text style={styles.cardDesc}>{item.desc}</Text>
      
      <View style={styles.metaBox}>
        <Text style={styles.metaText}><Text style={styles.bold}>Nhóm cơ:</Text> {item.muscle}</Text>
        <Text style={styles.metaText}><Text style={styles.bold}>Độ tuổi:</Text> {item.age}</Text>
        <Text style={styles.metaText}><Text style={styles.bold}>Calo/rep:</Text> {item.kcal}</Text>
      </View>
      
      <View style={{ alignItems: 'flex-end' }}>
        <TouchableOpacity style={styles.btnPrimary} onPress={() => router.push(`/workout/${item.slug}`)}>
          <Text style={styles.btnText}>Luyện tập</Text>
        </TouchableOpacity>
      </View>
    </View>
  );

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.eyebrow}>Thư viện động tác</Text>
        <Text style={styles.title}>Danh sách bài tập</Text>
        <View style={styles.searchBox}>
          <Ionicons name="search" size={20} color={Colors.muted} style={styles.searchIcon} />
          <TextInput 
            style={styles.searchInput}
            placeholder="Tìm theo tên bài tập, nhóm cơ..."
            placeholderTextColor={Colors.muted}
            value={search}
            onChangeText={setSearch} // Tự động cập nhật search state
          />
          {search.length > 0 && (
            <TouchableOpacity onPress={() => setSearch('')} style={styles.clearBtn}>
              <Ionicons name="close-circle" size={20} color={Colors.muted} />
            </TouchableOpacity>
          )}
        </View>
      </View>
      <FlatList
        data={filteredExercises}
        keyExtractor={(item) => item.id}
        renderItem={renderItem}
        contentContainerStyle={styles.listContent}
        ListEmptyComponent={
          <Text style={{ color: Colors.muted, textAlign: 'center', marginTop: 20 }}>Không tìm thấy bài tập nào.</Text>
        }
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.bg },
  header: { padding: 20, paddingTop: 60, backgroundColor: Colors.panel, borderBottomWidth: 1, borderBottomColor: Colors.line },
  eyebrow: { color: Colors.primary, textTransform: 'uppercase', fontWeight: 'bold', fontSize: 12, marginBottom: 8 },
  title: { fontSize: 26, color: Colors.text, fontWeight: 'bold', marginBottom: 16 },
  searchBox: { flexDirection: 'row', alignItems: 'center', backgroundColor: Colors.bg, borderWidth: 1, borderColor: Colors.line, borderRadius: 8, paddingHorizontal: 12, height: 44 },
  searchIcon: { marginRight: 8 },
  searchInput: { flex: 1, color: Colors.white, height: '100%' },
  clearBtn: { padding: 4 },
  listContent: { padding: 20, gap: 16 },
  card: { backgroundColor: Colors.panel, borderRadius: 12, padding: 20, borderWidth: 1, borderColor: Colors.line },
  cardHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 },
  cardTitle: { fontSize: 18, fontWeight: 'bold', color: Colors.text },
  diffPill: { borderWidth: 1, borderColor: Colors.primary, paddingHorizontal: 10, paddingVertical: 4, borderRadius: 6 },
  diffText: { color: Colors.primary, fontSize: 12, fontWeight: 'bold' },
  cardDesc: { color: Colors.text, fontSize: 14, marginBottom: 16, lineHeight: 20 },
  metaBox: { marginBottom: 16, gap: 4 },
  metaText: { color: Colors.text, fontSize: 13 },
  bold: { fontWeight: 'bold' },
  btnPrimary: { backgroundColor: Colors.primary, paddingHorizontal: 20, paddingVertical: 10, borderRadius: 6 },
  btnText: { color: Colors.white, fontWeight: 'bold', fontSize: 14 }
});
