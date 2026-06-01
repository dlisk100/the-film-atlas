export type FilmAtlasLayer = "macro" | "neighborhood" | "micro";

export interface FilmAtlasManifest {
  files: string[];
  generated_at?: string;
  layers?: Partial<Record<FilmAtlasLayer, {
    cluster_count?: number;
    labeled?: boolean;
  }>>;
  movie_count?: number;
  privacy?: {
    contains_api_keys?: boolean;
    contains_embeddings?: boolean;
    contains_raw_reviews?: boolean;
  };
  project?: string;
  projection_method?: string;
}

export interface FilmAtlasMovie {
  backdrop_path?: string | null;
  genres?: string[];
  imdb_id?: string | null;
  keywords?: string[];
  original_title?: string;
  overview?: string;
  popularity?: number;
  poster_path?: string | null;
  release_date?: string;
  runtime?: number | null;
  title?: string;
  tmdb_id: number;
  vote_average?: number;
  vote_count?: number;
  year?: number | null;
}

export interface FilmAtlasPoint {
  macro_id?: number | null;
  micro_id?: number | null;
  neighborhood_id?: number | null;
  tmdb_id: number;
  x: number;
  y: number;
}

export interface FilmAtlasLabel {
  cluster_id: number;
  confidence_score?: number;
  description?: string;
  label_id?: string;
  layer: FilmAtlasLayer;
  plain_label?: string;
  recommended_label?: string;
}

export interface FilmAtlasCluster {
  cluster_id: number;
  coherence_score?: number;
  description?: string;
  label_id?: string;
  parent_cluster_id?: number | null;
  recommended_label?: string;
  representative_movies?: string[];
  size?: number;
  terms?: string[];
  top_genres?: Array<[string, number]>;
  top_keywords?: Array<[string, number]>;
}

export interface FilmAtlasNeighbor {
  tmdb_id: number;
  title?: string;
  similarity?: number;
}

export interface FilmAtlasNeighborRecord {
  tmdb_id: number;
  neighbors?: FilmAtlasNeighbor[];
}

export interface FilmAtlasTerritoryPoint {
  tmdb_id: number;
  x: number;
  y: number;
}

export interface FilmAtlasTerritoryRegion {
  cluster_id: number;
  layer: FilmAtlasLayer;
  macro_id?: number | null;
  neighborhood_id?: number | null;
  parent_cluster_id?: number | null;
  radius: number;
  size?: number;
  x: number;
  y: number;
}

export interface FilmAtlasTerritoryVariant {
  algorithm?: string;
  description?: string;
  id: string;
  label: string;
  metrics?: {
    macro_regions?: number;
    micro_regions?: number;
    movie_points?: number;
    neighborhood_regions?: number;
  };
  points: FilmAtlasTerritoryPoint[];
  regions: FilmAtlasTerritoryRegion[];
}

export interface FilmAtlasTerritoryLayouts {
  generated_at?: string;
  movie_count?: number;
  source?: string;
  variants: FilmAtlasTerritoryVariant[];
}

export interface FilmAtlasExport {
  labels: FilmAtlasLabel[];
  macro_clusters: FilmAtlasCluster[];
  manifest: FilmAtlasManifest;
  micro_clusters: FilmAtlasCluster[];
  movies: FilmAtlasMovie[];
  neighborhood_clusters: FilmAtlasCluster[];
  neighbors: FilmAtlasNeighborRecord[];
  points: FilmAtlasPoint[];
}
