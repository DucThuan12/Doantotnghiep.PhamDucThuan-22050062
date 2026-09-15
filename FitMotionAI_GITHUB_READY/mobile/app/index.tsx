import { Redirect } from 'expo-router';

export default function IndexScreen() {
  // Tự động chuyển hướng người dùng vào trang Đăng nhập
  return <Redirect href="/(auth)/login" />;
}