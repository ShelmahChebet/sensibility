import { CameraView, CameraType, useCameraPermissions } from "expo-camera";
import { useState, useRef } from "react";
import { View, StyleSheet, Text, TouchableOpacity, Alert } from "react-native";

export default function SimpleCamera() {
  const [facing, setFacing] = useState<CameraType>("back");
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView>(null);

  if (!permission) return <View />; // still loading permissions

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.message}>
          We need your permission to show the camera
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

  const toggleCameraFacing = () => {
    setFacing((current) => (current === "back" ? "front" : "back"));
  };

  const takePicture = async () => {
    if (!cameraRef.current) return;
    const photo = await cameraRef.current.takePictureAsync(); // works here
    Alert.alert(
      "Photo taken!",
      "Do you want to use this photo?",
      [
        { text: "Cancel", style: "cancel" },
        { text: "Use Photo", onPress: () => console.log(photo.uri) },
      ],
      { cancelable: true },
    );
  };

  return (
    <View style={styles.container}>
      <CameraView
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        facing={facing}
      />
      <View style={styles.controls}>
        <TouchableOpacity onPress={takePicture} style={styles.snapButton}>
          <View style={styles.snapCircle} />
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "black" },
  message: {
    textAlign: "center",
    color: "white",
    fontSize: 16,
    marginBottom: 16,
  },
  permissionButton: {
    backgroundColor: "#fff",
    padding: 12,
    borderRadius: 10,
    alignSelf: "center",
  },
  permissionButtonText: { fontWeight: "bold", fontSize: 16 },
  controls: {
    position: "absolute",
    bottom: 50,
    width: "100%",
    flexDirection: "row",
    justifyContent: "space-around",
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
  buttonText: { fontWeight: "bold" },
});
