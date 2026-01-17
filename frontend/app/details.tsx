import { useLocalSearchParams } from "expo-router";
import { useEffect, useState } from "react";
import { ScrollView, Text, View, Image, StyleSheet } from "react-native";

export default function Details() {
  const params = useLocalSearchParams();

  useEffect(() => {}, []);

  return (
    <ScrollView contentContainerStyle={{ gap: 16, padding: 16 }}></ScrollView>
  );
}

const styles = StyleSheet.create({});
