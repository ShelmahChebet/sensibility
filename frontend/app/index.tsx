import { Link } from "expo-router";
import { useEffect, useState } from "react";
import {
  ScrollView,
  Text,
  View,
  Image,
  StyleSheet,
  Pressable,
} from "react-native";

interface Pokemon {
  name: string;
  image: string;
  imageBack: string; // Optional because it is added dynamically
  types: PokemonType[];
}

interface PokemonType {
  type: {
    name: string;
    url: string;
  };
}

const colorsByType = {
  grass: "#78C850",
  fire: "#F08030",
  water: "#6890F0",
  bug: "#A8B820",
  normal: "#A8A878",
  poison: "#A040A0",
  electric: "#F8D030",
  ground: "#E0C068",
  fairy: "#EE99AC",
  fighting: "#C03028",
  psychic: "#F85888",
  rock: "#B8A038",
  ghost: "#705898",
  ice: "#98D8D8",
  dragon: "#7038F8",
  dark: "#705848",
  steel: "#B8B8D0",
  flying: "#A890F0",
};

export default function Index() {
  const [pokemons, setPokemons] = useState<Pokemon[]>([]);

  useEffect(() => {
    // fetch pokemons
    fetchPokemons();
  }, []);

  async function fetchPokemons() {
    try {
      const response = await fetch(
        "https://pokeapi.co/api/v2/pokemon?limit=20",
      );
      const data = await response.json();

      // Fetch detailed info for each Pokemon in parallel
      const detailedPokemons = await Promise.all(
        data.results.map(async (pokemon: any) => {
          const res = await fetch(pokemon.url);
          const details = await res.json();
          return {
            name: pokemon.name,
            image: details.sprites.front_default, // main sprite
            imageBack: details.sprites.back_default, // back sprite
            types: details.types,
          };
        }),
      );

      setPokemons(detailedPokemons);
    } catch (e) {
      console.log(e);
    }
  }

  return (
    <ScrollView
      contentContainerStyle={{ gap: 16, padding: 16, alignItems: "center" }}
    >
      {pokemons.map((pokemon) => (
        <Link
          key={pokemon.name}
          href={{ pathname: "/wardrobe", params: { name: pokemon.name } }}
        >
          <View
            style={{
              // @ts-ignore
              backgroundColor: colorsByType[pokemon.types[0].type.name] + 50,
              padding: 20,
              borderRadius: 20,
            }}
          >
            <Text style={styles.name}>{pokemon.name}</Text>
            <Text style={styles.type}>{pokemon.types[0].type.name}</Text>

            <View style={{ flexDirection: "row", justifyContent: "center" }}>
              <Image
                source={{ uri: pokemon.image }}
                style={{ width: 100, height: 100 }}
              />
              <Image
                source={{ uri: pokemon.imageBack }}
                style={{ width: 100, height: 100 }}
              />
            </View>
          </View>
        </Link>
      ))}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  name: {
    fontSize: 28,
    fontWeight: "bold",
    textAlign: "center",
  },
  type: {
    fontSize: 20,
    fontWeight: "bold",
    color: "gray",
    textAlign: "center",
  },
});
