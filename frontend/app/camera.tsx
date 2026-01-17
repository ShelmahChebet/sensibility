import { useState, useRef } from "react";
import {
  View,
  StyleSheet,
  Text,
  TouchableOpacity,
  Image,
  SafeAreaView,
  ScrollView,
  Modal,
} from "react-native";
import { CameraView, CameraType, useCameraPermissions } from "expo-camera";
import { Ionicons } from "@expo/vector-icons";

type WardrobeItem = {
  id: string;
  uri: string;
  category?: string;
  addedDate: Date;
};

export default function WardrobeCamera() {
  const [permission, requestPermission] = useCameraPermissions();
  const [photo, setPhoto] = useState<string | null>(null);
  const [showCamera, setShowCamera] = useState(true);
  const [wardrobeItems, setWardrobeItems] = useState<WardrobeItem[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [showCategoryModal, setShowCategoryModal] = useState(false);

  const cameraRef = useRef<CameraView>(null);

  if (!permission) return <View />;

  if (!permission.granted) {
    return (
      <View style={styles.permissionContainer}>
        <Ionicons name="camera-outline" size={64} color="#666" />
        <Text style={styles.permissionTitle}>Camera Access Needed</Text>
        <Text style={styles.permissionText}>
          To add clothing items to your wardrobe, we need access to your camera.
        </Text>
        <TouchableOpacity
          onPress={requestPermission}
          style={styles.permissionButton}
        >
          <Text style={styles.permissionButtonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const takePicture = async () => {
    if (!cameraRef.current) return;
    try {
      const photoData = await cameraRef.current.takePictureAsync({
        quality: 0.8,
        base64: false,
      });
      setPhoto(photoData.uri);
      setShowCamera(false);
      setShowCategoryModal(true);
    } catch (error) {
      console.error("Error taking picture:", error);
    }
  };

  const addToWardrobe = () => {
    if (!photo) return;

    const newItem: WardrobeItem = {
      id: Date.now().toString(),
      uri: photo,
      category: selectedCategory || "Uncategorized",
      addedDate: new Date(),
    };

    setWardrobeItems((prev) => [...prev, newItem]);
    resetCamera();

    // Show success message
    setTimeout(() => {
      alert(`Added to ${selectedCategory || "your wardrobe"} successfully!`);
    }, 100);
  };

  const retakePhoto = () => {
    setPhoto(null);
    setSelectedCategory(null);
    setShowCategoryModal(false);
    setShowCamera(true);
  };

  const resetCamera = () => {
    setPhoto(null);
    setSelectedCategory(null);
    setShowCategoryModal(false);
    setShowCamera(true);
  };

  const clothingCategories = [
    { id: "top", name: "👕 Top", icon: "shirt-outline" },
    { id: "bottom", name: "👖 Bottom", icon: "body-outline" },
    { id: "dress", name: "👗 Dress", icon: "woman-outline" },
    { id: "outerwear", name: "🧥 Outerwear", icon: "cloudy-outline" },
    { id: "shoes", name: "👟 Shoes", icon: "walk-outline" },
    { id: "accessory", name: "👜 Accessory", icon: "bag-outline" },
  ];

  // Camera Screen
  if (showCamera) {
    return (
      <View style={styles.container}>
        <CameraView
          ref={cameraRef}
          style={StyleSheet.absoluteFill}
          facing="back"
        />

        {/* Camera overlay with guide */}
        <View style={styles.cameraOverlay}>
          <View style={styles.guideBox}>
            <View style={styles.guideLines}>
              <View style={styles.guideLineHorizontal} />
              <View style={styles.guideLineVertical} />
            </View>
          </View>

          <View style={styles.instructionContainer}>
            <Text style={styles.instructionTitle}>Add to Wardrobe</Text>
            <Text style={styles.instructionText}>
              Position clothing item within the frame
            </Text>
          </View>
        </View>

        {/* Camera controls */}
        <SafeAreaView style={styles.cameraControls}>
          <TouchableOpacity
            style={styles.closeButton}
            onPress={() => console.log("Navigate back")}
          >
            <Ionicons name="close" size={24} color="white" />
          </TouchableOpacity>

          <View style={styles.snapContainer}>
            <TouchableOpacity onPress={takePicture} style={styles.snapButton}>
              <View style={styles.snapCircle} />
            </TouchableOpacity>
          </View>

          <View style={styles.placeholder} />
        </SafeAreaView>
      </View>
    );
  }

  // Photo Confirmation Screen
  return (
    <SafeAreaView style={styles.confirmationContainer}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Text style={styles.headerTitle}>Add to Wardrobe</Text>
          <Text style={styles.headerSubtitle}>
            Confirm and categorize your item
          </Text>
        </View>

        {photo && (
          <View style={styles.photoContainer}>
            <Image source={{ uri: photo }} style={styles.photo} />
            <View style={styles.photoOverlay}>
              <TouchableOpacity style={styles.editButton} onPress={retakePhoto}>
                <Ionicons name="camera-reverse" size={20} color="white" />
                <Text style={styles.editButtonText}>Retake</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}

        <View style={styles.categorySection}>
          <Text style={styles.sectionTitle}>Category</Text>
          <Text style={styles.sectionSubtitle}>
            Select what type of clothing this is
          </Text>

          <View style={styles.categoryGrid}>
            {clothingCategories.map((category) => (
              <TouchableOpacity
                key={category.id}
                style={[
                  styles.categoryButton,
                  selectedCategory === category.id &&
                    styles.categoryButtonSelected,
                ]}
                onPress={() => setSelectedCategory(category.id)}
              >
                <Ionicons
                  name={category.icon as any}
                  size={24}
                  color={selectedCategory === category.id ? "#6366f1" : "#666"}
                />
                <Text
                  style={[
                    styles.categoryText,
                    selectedCategory === category.id &&
                      styles.categoryTextSelected,
                  ]}
                >
                  {category.name}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        <View style={styles.tipBox}>
          <Ionicons
            name="information-circle-outline"
            size={20}
            color="#6366f1"
          />
          <Text style={styles.tipText}>
            Adding categories helps us provide better outfit recommendations
          </Text>
        </View>
      </ScrollView>

      {/* Action buttons */}
      <View style={styles.actionButtons}>
        <TouchableOpacity style={styles.secondaryButton} onPress={retakePhoto}>
          <Text style={styles.secondaryButtonText}>Cancel</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[
            styles.primaryButton,
            (!selectedCategory || !photo) && styles.primaryButtonDisabled,
          ]}
          onPress={addToWardrobe}
          disabled={!selectedCategory || !photo}
        >
          <Ionicons name="add-circle-outline" size={20} color="white" />
          <Text style={styles.primaryButtonText}>Add to Wardrobe</Text>
        </TouchableOpacity>
      </View>

      {/* Category Selection Modal */}
      <Modal
        visible={showCategoryModal}
        transparent={true}
        animationType="slide"
        onRequestClose={() => setShowCategoryModal(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Categorize Item</Text>
            <Text style={styles.modalSubtitle}>
              Select a category for better organization
            </Text>

            <View style={styles.modalCategories}>
              {clothingCategories.map((category) => (
                <TouchableOpacity
                  key={category.id}
                  style={[
                    styles.modalCategoryButton,
                    selectedCategory === category.id &&
                      styles.modalCategoryButtonSelected,
                  ]}
                  onPress={() => {
                    setSelectedCategory(category.id);
                    setShowCategoryModal(false);
                  }}
                >
                  <Ionicons
                    name={category.icon as any}
                    size={24}
                    color={
                      selectedCategory === category.id ? "#6366f1" : "#666"
                    }
                  />
                  <Text style={styles.modalCategoryText}>{category.name}</Text>
                </TouchableOpacity>
              ))}
            </View>

            <TouchableOpacity
              style={styles.skipButton}
              onPress={() => {
                setSelectedCategory("uncategorized");
                setShowCategoryModal(false);
              }}
            >
              <Text style={styles.skipButtonText}>Skip for now</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  // Permission Screen
  permissionContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#f8f9fa",
    padding: 24,
  },
  permissionTitle: {
    fontSize: 24,
    fontWeight: "700",
    color: "#333",
    marginTop: 16,
    marginBottom: 8,
  },
  permissionText: {
    fontSize: 16,
    color: "#666",
    textAlign: "center",
    lineHeight: 22,
    marginBottom: 32,
  },
  permissionButton: {
    backgroundColor: "#6366f1",
    paddingHorizontal: 24,
    paddingVertical: 14,
    borderRadius: 12,
  },
  permissionButtonText: {
    color: "white",
    fontSize: 16,
    fontWeight: "600",
  },

  // Camera Screen
  container: {
    flex: 1,
    backgroundColor: "black",
  },
  cameraOverlay: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  guideBox: {
    width: 300,
    height: 400,
    borderWidth: 2,
    borderColor: "rgba(255, 255, 255, 0.5)",
    borderRadius: 12,
    overflow: "hidden",
  },
  guideLines: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  guideLineHorizontal: {
    width: "100%",
    height: 1,
    backgroundColor: "rgba(255, 255, 255, 0.3)",
  },
  guideLineVertical: {
    height: "100%",
    width: 1,
    backgroundColor: "rgba(255, 255, 255, 0.3)",
  },
  instructionContainer: {
    position: "absolute",
    top: 100,
    alignItems: "center",
  },
  instructionTitle: {
    color: "white",
    fontSize: 20,
    fontWeight: "600",
    textShadowColor: "rgba(0, 0, 0, 0.75)",
    textShadowOffset: { width: 1, height: 1 },
    textShadowRadius: 3,
  },
  instructionText: {
    color: "rgba(255, 255, 255, 0.8)",
    fontSize: 14,
    marginTop: 4,
    textShadowColor: "rgba(0, 0, 0, 0.75)",
    textShadowOffset: { width: 1, height: 1 },
    textShadowRadius: 3,
  },
  cameraControls: {
    position: "absolute",
    bottom: 0,
    left: 0,
    right: 0,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: 20,
    paddingBottom: 40,
  },
  closeButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: "rgba(0, 0, 0, 0.5)",
    justifyContent: "center",
    alignItems: "center",
  },
  snapContainer: {
    flex: 1,
    alignItems: "center",
  },
  snapButton: {
    width: 70,
    height: 70,
    borderRadius: 35,
    borderWidth: 4,
    borderColor: "white",
    justifyContent: "center",
    alignItems: "center",
  },
  snapCircle: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: "white",
  },
  placeholder: {
    width: 44,
  },

  // Confirmation Screen
  confirmationContainer: {
    flex: 1,
    backgroundColor: "#fff",
  },
  scrollContent: {
    flexGrow: 1,
    paddingBottom: 100,
  },
  header: {
    padding: 24,
    paddingTop: 16,
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: "700",
    color: "#333",
    marginBottom: 4,
  },
  headerSubtitle: {
    fontSize: 16,
    color: "#666",
  },
  photoContainer: {
    marginHorizontal: 24,
    marginBottom: 24,
    borderRadius: 16,
    overflow: "hidden",
    backgroundColor: "#f5f5f5",
  },
  photo: {
    width: "100%",
    height: 400,
    transform: [{ scaleX: -1 }],
  },
  photoOverlay: {
    position: "absolute",
    top: 12,
    right: 12,
  },
  editButton: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "rgba(0, 0, 0, 0.6)",
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 20,
    gap: 6,
  },
  editButtonText: {
    color: "white",
    fontSize: 14,
    fontWeight: "500",
  },
  categorySection: {
    paddingHorizontal: 24,
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: "600",
    color: "#333",
    marginBottom: 4,
  },
  sectionSubtitle: {
    fontSize: 14,
    color: "#666",
    marginBottom: 16,
  },
  categoryGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 12,
  },
  categoryButton: {
    flex: 1,
    minWidth: "30%",
    alignItems: "center",
    padding: 16,
    backgroundColor: "#f8f9fa",
    borderRadius: 12,
    borderWidth: 2,
    borderColor: "transparent",
  },
  categoryButtonSelected: {
    backgroundColor: "#f0f0ff",
    borderColor: "#6366f1",
  },
  categoryText: {
    marginTop: 8,
    fontSize: 12,
    color: "#666",
    textAlign: "center",
  },
  categoryTextSelected: {
    color: "#6366f1",
    fontWeight: "600",
  },
  tipBox: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#f0f0ff",
    marginHorizontal: 24,
    padding: 16,
    borderRadius: 12,
    gap: 12,
  },
  tipText: {
    flex: 1,
    fontSize: 14,
    color: "#4f46e5",
    lineHeight: 20,
  },
  actionButtons: {
    position: "absolute",
    bottom: 0,
    left: 0,
    right: 0,
    flexDirection: "row",
    padding: 16,
    backgroundColor: "white",
    borderTopWidth: 1,
    borderTopColor: "#e5e5e5",
    gap: 12,
  },
  secondaryButton: {
    flex: 1,
    padding: 16,
    alignItems: "center",
    backgroundColor: "#f5f5f5",
    borderRadius: 12,
  },
  secondaryButtonText: {
    fontSize: 16,
    fontWeight: "600",
    color: "#666",
  },
  primaryButton: {
    flex: 2,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    padding: 16,
    backgroundColor: "#6366f1",
    borderRadius: 12,
    gap: 8,
  },
  primaryButtonDisabled: {
    backgroundColor: "#c7d2fe",
  },
  primaryButtonText: {
    fontSize: 16,
    fontWeight: "600",
    color: "white",
  },

  // Modal
  modalOverlay: {
    flex: 1,
    backgroundColor: "rgba(0, 0, 0, 0.5)",
    justifyContent: "flex-end",
  },
  modalContent: {
    backgroundColor: "white",
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    padding: 24,
    paddingBottom: 40,
  },
  modalTitle: {
    fontSize: 22,
    fontWeight: "700",
    color: "#333",
    marginBottom: 8,
  },
  modalSubtitle: {
    fontSize: 14,
    color: "#666",
    marginBottom: 24,
  },
  modalCategories: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 12,
    marginBottom: 24,
  },
  modalCategoryButton: {
    flexDirection: "row",
    alignItems: "center",
    flex: 1,
    minWidth: "45%",
    padding: 16,
    backgroundColor: "#f8f9fa",
    borderRadius: 12,
    borderWidth: 2,
    borderColor: "transparent",
    gap: 12,
  },
  modalCategoryButtonSelected: {
    backgroundColor: "#f0f0ff",
    borderColor: "#6366f1",
  },
  modalCategoryText: {
    fontSize: 14,
    color: "#333",
    fontWeight: "500",
  },
  skipButton: {
    padding: 16,
    alignItems: "center",
  },
  skipButtonText: {
    fontSize: 16,
    color: "#666",
    fontWeight: "500",
  },
});
