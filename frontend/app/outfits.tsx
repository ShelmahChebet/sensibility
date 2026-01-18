import React, { useState } from "react";
import { View } from "react-native";
import OutfitPrompt from "../components/OutfitPrompt";
import OutfitResults from "../components/OutfitResults";

export default function OutfitsScreen() {
  const [showResults, setShowResults] = useState(false);
  const [outfitData, setOutfitData] = useState<any>(null);

  const handleOutfitsGenerated = (data: any) => {
    setOutfitData(data);
    setShowResults(true);
  };

  const handleBackToPrompt = () => {
    setShowResults(false);
  };

  return (
    <View style={{ flex: 1, backgroundColor: "#fff" }}>
      {!showResults ? (
        <OutfitPrompt
          userId="test_user"
          onOutfitsGenerated={handleOutfitsGenerated}
        />
      ) : (
        <OutfitResults
          userId="test_user"
          outfitData={outfitData}
          onBack={handleBackToPrompt}
        />
      )}
    </View>
  );
}