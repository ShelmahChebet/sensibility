import React, { useState } from "react";
import {
  View,
  Text,
  ScrollView,
  Image,
  TouchableOpacity,
  StyleSheet,
  Alert,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import Constants from "expo-constants";

const API_URL = Constants.expoConfig?.extra?.apiUrl || "http://138.51.75.132:8000";

interface OutfitItem {
  filename: string;
  category: string;
  style: string;
  image_url?: string;
}

interface Outfit {
  outfit_number: number;
  type: string;
  items: OutfitItem[];
  scores: {
    compatibility: number;
    prompt_match: number;
    overall: number;
  };
}

interface OutfitResultsProps {
  userId?: string;
  outfitData: {
    prompt: string;
    outfits: Outfit[];
    explanation: {
      explanation: string;
      top_reason: string;
      styling_tips?: string;
    };
  };
  onBack?: () => void;
}

export default function OutfitResults({
  userId = "test_user",
  outfitData,
  onBack,
}: OutfitResultsProps) {
  const [ratings, setRatings] = useState<{ [key: number]: number }>({});
  const [submittingRating, setSubmittingRating] = useState<number | null>(null);

  const handleRating = async (outfitNumber: number, rating: number) => {
    setRatings((prev) => ({ ...prev, [outfitNumber]: rating }));
    setSubmittingRating(outfitNumber);

    const outfit = outfitData.outfits.find((o) => o.outfit_number === outfitNumber);
    if (!outfit) return;

    try {
      const response = await fetch(`${API_URL}/rate_outfit`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          user_id: userId,
          outfit_number: outfitNumber,
          rating: rating,
          items: outfit.items.map((item) => item.filename),
          outfit_type: outfit.type,
        }),
      });

      const result = await response.json();

      if (response.ok) {
        Alert.alert(
          "Rating Saved! ⭐",
          result.message || "Thanks for your feedback!",
          [{ text: "OK" }]
        );
      } else {
        Alert.alert("Error", result.error || "Failed to save rating");
      }
    } catch (err) {
      console.error("Error submitting rating:", err);
      Alert.alert("Error", "Failed to save rating. Please try again.");
    } finally {
      setSubmittingRating(null);
    }
  };

  const StarRating = ({
    outfitNumber,
    currentRating,
  }: {
    outfitNumber: number;
    currentRating: number;
  }) => {
    return (
      <View style={styles.starContainer}>
        <Text style={styles.rateLabel}>Rate this outfit:</Text>
        <View style={styles.stars}>
          {[1, 2, 3, 4, 5].map((star) => (
            <TouchableOpacity
              key={star}
              onPress={() => handleRating(outfitNumber, star)}
              disabled={submittingRating === outfitNumber}
            >
              <Text style={styles.star}>
                {star <= currentRating ? "⭐" : "☆"}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>
    );
  };

  return (
    <SafeAreaView style={styles.safeArea} edges={['top']}>
      <View style={styles.container}>
        <View style={styles.header}>
          {onBack && (
            <TouchableOpacity onPress={onBack} style={styles.backButton}>
              <Text style={styles.backButtonText}>← Back</Text>
            </TouchableOpacity>
          )}
          <Text style={styles.title}>Your Outfit Suggestions</Text>
          <Text style={styles.promptText}>{`"${outfitData.prompt}"`}</Text>
        </View>

        <ScrollView contentContainerStyle={styles.scrollContent}>
          <View style={styles.explanationCard}>
            <Text style={styles.explanationTitle}>💡 Why these outfits?</Text>
            <Text style={styles.explanationText}>
              {outfitData.explanation.explanation}
            </Text>
            {outfitData.explanation.styling_tips && (
              <>
                <Text style={styles.tipsTitle}>✨ Styling Tips:</Text>
                <Text style={styles.tipsText}>
                  {outfitData.explanation.styling_tips}
                </Text>
              </>
            )}
          </View>

          {outfitData.outfits.map((outfit) => (
            <View key={outfit.outfit_number} style={styles.outfitCard}>
              <View style={styles.outfitHeader}>
                <Text style={styles.outfitNumber}>
                  Outfit #{outfit.outfit_number}
                </Text>
                <View style={styles.badge}>
                  <Text style={styles.badgeText}>
                    {outfit.type === "dress" ? "👗 Dress" : "👔 Separates"}
                  </Text>
                </View>
              </View>

              <ScrollView
                horizontal
                showsHorizontalScrollIndicator={false}
                style={styles.itemsScroll}
              >
                {outfit.items.map((item, index) => (
                  <View key={index} style={styles.itemCard}>
                    {item.image_url ? (
                      <Image
                        source={{ uri: item.image_url }}
                        style={styles.itemImage}
                        resizeMode="cover"
                      />
                    ) : (
                      <View style={[styles.itemImage, styles.placeholderImage]}>
                        <Text style={styles.placeholderText}>📷</Text>
                      </View>
                    )}
                    <View style={styles.itemInfo}>
                      <Text style={styles.itemCategory}>{item.category}</Text>
                      {item.style !== "unknown" && (
                        <Text style={styles.itemStyle}>{item.style}</Text>
                      )}
                    </View>
                  </View>
                ))}
              </ScrollView>

              <View style={styles.scoresContainer}>
                <View style={styles.scoreItem}>
                  <Text style={styles.scoreLabel}>Match</Text>
                  <Text style={styles.scoreValue}>
                    {outfit.scores.prompt_match}%
                  </Text>
                </View>
                <View style={styles.scoreItem}>
                  <Text style={styles.scoreLabel}>Compatibility</Text>
                  <Text style={styles.scoreValue}>
                    {outfit.scores.compatibility}%
                  </Text>
                </View>
                <View style={styles.scoreItem}>
                  <Text style={styles.scoreLabel}>Overall</Text>
                  <Text style={[styles.scoreValue, styles.scoreValuePrimary]}>
                    {outfit.scores.overall}%
                  </Text>
                </View>
              </View>

              <StarRating
                outfitNumber={outfit.outfit_number}
                currentRating={ratings[outfit.outfit_number] || 0}
              />
            </View>
          ))}

          <View style={{ height: 100 }} />
        </ScrollView>
      </View>
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
  header: {
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: "#e0e0e0",
    backgroundColor: "#fff",
  },
  backButton: {
    marginBottom: 8,
  },
  backButtonText: {
    fontSize: 16,
    color: "#4B9CE2",
    fontWeight: "500",
  },
  title: {
    fontSize: 24,
    fontWeight: "bold",
    color: "#333",
    marginBottom: 4,
  },
  promptText: {
    fontSize: 14,
    color: "#666",
    fontStyle: "italic",
  },
  scrollContent: {
    padding: 16,
  },
  explanationCard: {
    backgroundColor: "#f0f8ff",
    padding: 16,
    borderRadius: 12,
    marginBottom: 20,
  },
  explanationTitle: {
    fontSize: 16,
    fontWeight: "bold",
    color: "#333",
    marginBottom: 8,
  },
  explanationText: {
    fontSize: 14,
    color: "#555",
    lineHeight: 20,
  },
  tipsTitle: {
    fontSize: 14,
    fontWeight: "600",
    color: "#333",
    marginTop: 12,
    marginBottom: 4,
  },
  tipsText: {
    fontSize: 13,
    color: "#555",
    lineHeight: 18,
    fontStyle: "italic",
  },
  outfitCard: {
    backgroundColor: "#fff",
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: "#e0e0e0",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  outfitHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 12,
  },
  outfitNumber: {
    fontSize: 18,
    fontWeight: "bold",
    color: "#333",
  },
  badge: {
    backgroundColor: "#e8f4ff",
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
  },
  badgeText: {
    fontSize: 12,
    color: "#4B9CE2",
    fontWeight: "600",
  },
  itemsScroll: {
    marginBottom: 16,
  },
  itemCard: {
    width: 140,
    marginRight: 12,
    borderRadius: 8,
    overflow: "hidden",
    backgroundColor: "#f9f9f9",
    borderWidth: 1,
    borderColor: "#e0e0e0",
  },
  itemImage: {
    width: "100%",
    height: 140,
    backgroundColor: "#e8e8e8",
  },
  placeholderImage: {
    justifyContent: "center",
    alignItems: "center",
  },
  placeholderText: {
    fontSize: 48,
  },
  itemInfo: {
    padding: 8,
  },
  itemCategory: {
    fontSize: 12,
    fontWeight: "600",
    color: "#333",
    textTransform: "capitalize",
  },
  itemStyle: {
    fontSize: 10,
    color: "#666",
    marginTop: 2,
    fontStyle: "italic",
  },
  scoresContainer: {
    flexDirection: "row",
    justifyContent: "space-around",
    paddingVertical: 12,
    borderTopWidth: 1,
    borderBottomWidth: 1,
    borderColor: "#f0f0f0",
    marginBottom: 12,
  },
  scoreItem: {
    alignItems: "center",
  },
  scoreLabel: {
    fontSize: 11,
    color: "#999",
    marginBottom: 4,
  },
  scoreValue: {
    fontSize: 16,
    fontWeight: "bold",
    color: "#555",
  },
  scoreValuePrimary: {
    color: "#4B9CE2",
  },
  starContainer: {
    alignItems: "center",
    paddingTop: 8,
  },
  rateLabel: {
    fontSize: 12,
    color: "#666",
    marginBottom: 6,
  },
  stars: {
    flexDirection: "row",
    gap: 8,
  },
  star: {
    fontSize: 28,
  },
});