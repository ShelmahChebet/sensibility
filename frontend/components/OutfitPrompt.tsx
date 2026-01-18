import React, { useState } from "react";
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import Constants from "expo-constants";

const API_URL = Constants.expoConfig?.extra?.apiUrl || "http://138.51.75.132:8000";

interface OutfitPromptProps {
  userId?: string;
  onOutfitsGenerated?: (data: any) => void;
}

export default function OutfitPrompt({
  userId = "test_user",
  onOutfitsGenerated,
}: OutfitPromptProps) {
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const suggestions = [
    "office outfit",
    "casual weekend",
    "date night",
    "beach vacation",
    "gym workout",
    "formal event",
    "brunch with friends",
    "job interview",
  ];

  const handleSuggestionPress = (suggestion: string) => {
    setPrompt(suggestion);
  };

  const handleGenerateOutfits = async () => {
    if (!prompt.trim()) {
      setError("Please enter what you're looking for!");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(
      `${API_URL}/suggest_outfits?prompt=${encodeURIComponent(
        prompt
        )}&user_id=${userId}&num_outfits=3`,
        {
          method: "POST",
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (data.error) {
        setError(data.message || "Failed to generate outfits");
      } else {
        if (onOutfitsGenerated) {
          onOutfitsGenerated(data);
        }
      }
    } catch (err) {
      console.error("Error generating outfits:", err);
      setError(
        err instanceof Error ? err.message : "Failed to generate outfits"
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.safeArea} edges={['top']}>
      <KeyboardAvoidingView
        behavior={Platform.OS === "ios" ? "padding" : "height"}
        style={styles.container}
      >
        <ScrollView
          contentContainerStyle={styles.scrollContent}
          keyboardShouldPersistTaps="handled"
        >
          <View style={styles.header}>
            <Text style={styles.title}>What are you looking for?</Text>
            <Text style={styles.subtitle}>
              Describe the outfit you need and we will create perfect combinations
              from your wardrobe
            </Text>
          </View>

          <View style={styles.inputContainer}>
            <TextInput
              style={styles.input}
              placeholder="E.g., 'smart casual for work meeting'"
              placeholderTextColor="#999"
              value={prompt}
              onChangeText={(text) => {
                setPrompt(text);
                setError(null);
              }}
              multiline
              numberOfLines={3}
              maxLength={200}
              autoCapitalize="sentences"
            />
            <Text style={styles.charCount}>{prompt.length}/200</Text>
          </View>

          <View style={styles.suggestionsContainer}>
            <Text style={styles.suggestionsTitle}>Quick ideas:</Text>
            <View style={styles.suggestionsGrid}>
              {suggestions.map((suggestion, index) => (
                <TouchableOpacity
                  key={index}
                  style={styles.suggestionChip}
                  onPress={() => handleSuggestionPress(suggestion)}
                >
                  <Text style={styles.suggestionText}>{suggestion}</Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>

          {error && (
            <View style={styles.errorContainer}>
              <Text style={styles.errorText}>⚠️ {error}</Text>
            </View>
          )}

          <TouchableOpacity
            style={[
              styles.generateButton,
              (!prompt.trim() || loading) && styles.generateButtonDisabled,
            ]}
            onPress={handleGenerateOutfits}
            disabled={!prompt.trim() || loading}
          >
            {loading ? (
              <View style={styles.loadingContainer}>
                <ActivityIndicator color="#fff" size="small" />
                <Text style={styles.generateButtonText}>
                  Creating outfits...
                </Text>
              </View>
            ) : (
              <Text style={styles.generateButtonText}>✨ Generate Outfits</Text>
            )}
          </TouchableOpacity>

          <Text style={styles.infoText}>
            We will suggest 3 complete outfit combinations based on your request
            and style preferences
          </Text>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: "#fff",
  },
  container: {
    flex: 1,
  },
  scrollContent: {
    padding: 20,
    paddingBottom: 100,
  },
  header: {
    marginBottom: 24,
  },
  title: {
    fontSize: 28,
    fontWeight: "bold",
    color: "#333",
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: "#666",
    lineHeight: 22,
  },
  inputContainer: {
    marginBottom: 24,
  },
  input: {
    backgroundColor: "#f9f9f9",
    borderRadius: 12,
    padding: 16,
    fontSize: 16,
    color: "#333",
    borderWidth: 1,
    borderColor: "#e0e0e0",
    minHeight: 100,
    textAlignVertical: "top",
  },
  charCount: {
    fontSize: 12,
    color: "#999",
    textAlign: "right",
    marginTop: 4,
  },
  suggestionsContainer: {
    marginBottom: 24,
  },
  suggestionsTitle: {
    fontSize: 14,
    fontWeight: "600",
    color: "#666",
    marginBottom: 12,
  },
  suggestionsGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
  },
  suggestionChip: {
    backgroundColor: "#f0f0f0",
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 20,
    marginRight: 8,
    marginBottom: 8,
  },
  suggestionText: {
    fontSize: 14,
    color: "#555",
  },
  errorContainer: {
    backgroundColor: "#ffe6e6",
    padding: 12,
    borderRadius: 8,
    marginBottom: 16,
  },
  errorText: {
    color: "#d32f2f",
    fontSize: 14,
    textAlign: "center",
  },
  generateButton: {
    backgroundColor: "#4B9CE2",
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: "center",
    marginBottom: 12,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  generateButtonDisabled: {
    backgroundColor: "#ccc",
    shadowOpacity: 0,
    elevation: 0,
  },
  loadingContainer: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  generateButtonText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "600",
  },
  infoText: {
    fontSize: 12,
    color: "#999",
    textAlign: "center",
    fontStyle: "italic",
  },
});