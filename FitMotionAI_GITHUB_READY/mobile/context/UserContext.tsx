import React, { createContext, useState, useContext } from 'react';

// Cấu trúc dữ liệu người dùng
type UserContextType = {
  health: string;
  setHealth: (val: string) => void;
  goal: string;
  setGoal: (val: string) => void;
};

// Tạo Context
const UserContext = createContext<UserContextType | undefined>(undefined);

// Provider bọc toàn app
export function UserProvider({ children }: { children: React.ReactNode }) {
  const [health, setHealth] = useState("Huyết áp không ổn định");
  const [goal, setGoal] = useState("Tăng cơ");

  return (
    <UserContext.Provider value={{ health, setHealth, goal, setGoal }}>
      {children}
    </UserContext.Provider>
  );
}

// Hook để các trang khác gọi ra xài
export const useUser = () => {
  const context = useContext(UserContext);
  if (!context) throw new Error('useUser must be used within a UserProvider');
  return context;
};