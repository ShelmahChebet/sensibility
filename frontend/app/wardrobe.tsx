import React, { useEffect, useState } from "react";
import { View, Text, ScrollView, Image, TouchableOpacity, StyleSheet, ActivityIndicator, Alert, RefreshControl } from "react-native";
import Constants from 'expo-constants';

// Get API URL from app.json extra config or use default
const API_URL = Constants.expoConfig?.extra?.apiUrl || 'http://138.51.75.132:8000';

interface WardrobeItem {
  filename: string;
  category: string;
  style?: string;
  image_url: string;
  user_id: string;
}

export default function Wardrobe({ userId = "test_user" }: { userId?: string }) {
  const [clothes, setClothes] = useState<WardrobeItem[]>([]);
  const [categories, setCategories] = useState<string[]>(["All"]);
  const [selectedCategory, setSelectedCategory] = useState<string>("All");
  const [loading, setLoading] = useState<boolean>(true);
  const [refreshing, setRefreshing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const BACKEND_URL = API_URL;

  console.log("=== WARDROBE DEBUG ===");
  console.log("API_URL:", API_URL);
  console.log("BACKEND_URL:", BACKEND_URL);
  console.log("Full URL will be:", `${BACKEND_URL}/wardrobe_items/${userId}`);
  console.log("====================");

  const fetchWardrobe = async () => {
    try {
      console.log(`Fetching wardrobe for user: ${userId}`);
      
      const res = await fetch(`${BACKEND_URL}/wardrobe_items/${userId}`);
      
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      
      const data = await res.json();
      console.log("Received wardrobe data:", data);

      if (Array.isArray(data) && data.length > 0) {
      console.log("First item image_url:", data[0].image_url);
      console.log("Sample image URLs:", data.slice(0, 3).map(item => item.image_url));
     }

      if (Array.isArray(data)) {
        setClothes(data);
        
        // Extract unique categories
        const uniqueCategories = Array.from(
          new Set(data.map((item: WardrobeItem) => item.category))
        ) as string[];
        setCategories(["All", ...uniqueCategories]);
        setError(null);
      } else {
        console.error("Unexpected data format:", data);
        setError("Unexpected data format received");
      }
    } catch (err) {
      console.error("Error fetching wardrobe:", err);
      setError(err instanceof Error ? err.message : "Failed to load wardrobe");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchWardrobe();
  }, [userId]);

  const onRefresh = () => {
    setRefreshing(true);
    fetchWardrobe();
  };

  const handleDeleteItem = async (filename: string) => {
    Alert.alert(
      "Delete Item",
      "Are you sure you want to delete this item?",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Delete",
          style: "destructive",
          onPress: async () => {
            try {
              const res = await fetch(`${BACKEND_URL}/wardrobe_item/${userId}/${filename}`, {
                method: "DELETE",
              });

              if (res.ok) {
                // Remove from local state
                setClothes(prev => prev.filter(item => item.filename !== filename));
                Alert.alert("Success", "Item deleted successfully");
              } else {
                Alert.alert("Error", "Failed to delete item");
              }
            } catch (err) {
              console.error("Error deleting item:", err);
              Alert.alert("Error", "Failed to delete item");
            }
          },
        },
      ]
    );
  };

  // Filter clothes by selected category
  const filteredClothes =
    selectedCategory === "All"
      ? clothes
      : clothes.filter(item => item.category === selectedCategory);

  return (
    <View style={styles.container}>
      <Text style={styles.title}>My Wardrobe</Text>
      <Text style={styles.subtitle}>{clothes.length} items</Text>

      {/* Category Filters */}
      <ScrollView 
        horizontal 
        showsHorizontalScrollIndicator={false} 
        style={styles.categoryBar}
      >
        {categories.map(cat => (
          <TouchableOpacity
            key={cat}
            style={[
              styles.categoryButton,
              selectedCategory === cat && styles.categoryButtonActive,
            ]}
            onPress={() => setSelectedCategory(cat)}
          >
            <Text
              style={[
                styles.categoryText,
                selectedCategory === cat && styles.categoryTextActive,
              ]}
            >
              {cat}
            </Text>
          </TouchableOpacity>
        ))}
      </ScrollView>

      {/* Loading State */}
      {loading ? (
        <View style={styles.centerContainer}>
          <ActivityIndicator size="large" color="#4B9CE2" />
          <Text style={styles.loadingText}>Loading your wardrobe...</Text>
        </View>
      ) : error ? (
        /* Error State */
        <View style={styles.centerContainer}>
          <Text style={styles.errorText}>⚠️ {error}</Text>
          <TouchableOpacity 
            style={styles.retryButton}
            onPress={fetchWardrobe}
          >
            <Text style={styles.retryButtonText}>Retry</Text>
          </TouchableOpacity>
        </View>
      ) : filteredClothes.length === 0 ? (
        /* Empty State */
        <View style={styles.centerContainer}>
          <Text style={styles.emptyText}>
            {selectedCategory === "All" 
              ? "No items in your wardrobe yet.\nStart by uploading some clothes!" 
              : `No ${selectedCategory} items found.`}
          </Text>
        </View>
      ) : (
        /* Wardrobe Items Grid */
        <ScrollView 
          contentContainerStyle={styles.grid}
          refreshControl={
            <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
          }
        >
          {filteredClothes.map((item, index) => (
            <View key={`${item.filename}-${index}`} style={styles.card}>
              {/* Real Image from Supabase Storage */}
              <Image
                source={{ uri: item.image_url }}
                style={styles.image}
                resizeMode="cover"
                onError={(e) => {
                  console.error("Image load error:", e.nativeEvent.error);
                }}
              />
              
              <View style={styles.cardContent}>
                <Text style={styles.categoryLabel}>{item.category}</Text>
                {item.style && item.style !== "unknown" && (
                  <Text style={styles.styleLabel}>{item.style}</Text>
                )}
              </View>

              {/* Delete Button */}
              <TouchableOpacity
                style={styles.deleteButton}
                onPress={() => handleDeleteItem(item.filename)}
              >
                <Text style={styles.deleteButtonText}>🗑️</Text>
              </TouchableOpacity>
            </View>
          ))}
        </ScrollView>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
    backgroundColor: "#fff",
  },
  title: {
    fontSize: 28,
    fontWeight: "bold",
    marginBottom: 4,
    color: "#333",
  },
  subtitle: {
    fontSize: 14,
    color: "#666",
    marginBottom: 16,
  },
  categoryBar: {
    flexDirection: "row",
    marginBottom: 16,
    maxHeight: 40,
  },
  categoryButton: {
    paddingVertical: 8,
    paddingHorizontal: 16,
    backgroundColor: "#f0f0f0",
    borderRadius: 20,
    marginRight: 8,
    height: 36,
    justifyContent: "center",
  },
  categoryButtonActive: {
    backgroundColor: "#4B9CE2",
  },
  categoryText: {
    fontSize: 14,
    color: "#555",
    fontWeight: "500",
  },
  categoryTextActive: {
    color: "#fff",
    fontWeight: "600",
  },
  grid: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "space-between",
    paddingBottom: 16,
  },
  card: {
    width: "48%",
    marginBottom: 16,
    borderRadius: 12,
    overflow: "hidden",
    backgroundColor: "#f9f9f9",
    borderWidth: 1,
    borderColor: "#e0e0e0",
    position: "relative",
  },
  image: {
    width: "100%",
    height: 180,
    backgroundColor: "#e8e8e8",
  },
  cardContent: {
    padding: 8,
  },
  categoryLabel: {
    fontSize: 14,
    fontWeight: "600",
    color: "#333",
    textTransform: "capitalize",
  },
  styleLabel: {
    fontSize: 12,
    color: "#666",
    marginTop: 2,
    fontStyle: "italic",
  },
  deleteButton: {
    position: "absolute",
    top: 8,
    right: 8,
    backgroundColor: "rgba(255, 255, 255, 0.9)",
    borderRadius: 16,
    width: 32,
    height: 32,
    justifyContent: "center",
    alignItems: "center",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  deleteButtonText: {
    fontSize: 16,
  },
  centerContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    paddingTop: 60,
  },
  loadingText: {
    marginTop: 12,
    fontSize: 14,
    color: "#666",
  },
  emptyText: {
    textAlign: "center",
    fontSize: 16,
    color: "#888",
    lineHeight: 24,
  },
  errorText: {
    textAlign: "center",
    fontSize: 16,
    color: "#d32f2f",
    marginBottom: 16,
  },
  retryButton: {
    backgroundColor: "#4B9CE2",
    paddingVertical: 10,
    paddingHorizontal: 24,
    borderRadius: 8,
  },
  retryButtonText: {
    color: "#fff",
    fontSize: 14,
    fontWeight: "600",
  },
});