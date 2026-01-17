import { Tabs } from "expo-router";
import { Ionicons } from "@expo/vector-icons";

export default function RootLayout() {
  return (
    <Tabs
      screenOptions={({ route }) => ({
        tabBarIcon: ({ focused, color, size }) => {
          let iconName: keyof typeof Ionicons.glyphMap = "home";

          if (route.name === "index")
            iconName = focused ? "home" : "home-outline";
          if (route.name === "camera")
            iconName = focused ? "camera" : "camera-outline";
          if (route.name === "wardrobe")
            iconName = focused ? "shirt" : "shirt-outline";

          return <Ionicons name={iconName} size={size} color={color} />;
        },
        tabBarActiveTintColor: "#000",
        tabBarInactiveTintColor: "#999",
        tabBarStyle: {
          backgroundColor: "#fff",
          borderTopLeftRadius: 20,
          borderTopRightRadius: 20,
          height: 70,
          position: "absolute",
        },
        tabBarLabelStyle: {
          fontSize: 12,
          marginBottom: 5,
        },
        headerShown: true,
      })}
    >
      {/* Set custom title for the Home tab */}
      <Tabs.Screen
        name="index"
        options={{ title: "Home" }} // ✅ This changes the top bar text
      />
      <Tabs.Screen name="camera" options={{ title: "Camera" }} />
      <Tabs.Screen name="wardrobe" options={{ title: "Wardrobe" }} />
    </Tabs>
  );
}
