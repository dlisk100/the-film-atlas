import type {
  FilmAtlasCluster,
  FilmAtlasExport,
  FilmAtlasGmapCell,
  FilmAtlasLabel,
  FilmAtlasLayer,
  FilmAtlasMovie,
  FilmAtlasNeighborRecord,
  FilmAtlasPoint,
  FilmAtlasTerritoryLayouts,
  FilmAtlasTerritoryRegion,
  FilmAtlasTerritoryVariant,
} from "./types";

type AtlasNode = {
  color: string;
  macroId: number | null;
  macroLabel: string;
  mainLabel: string;
  microId: number | null;
  microLabel: string;
  movie: FilmAtlasMovie;
  neighborhoodId: number | null;
  neighborhoodLabel: string;
  point: FilmAtlasPoint;
  searchText: string;
};

type ClusterLabel = {
  color: string;
  count: number;
  id: number;
  label: string;
  layer: FilmAtlasLayer;
  x: number;
  y: number;
};

type ClusterLabelPlacement = {
  heightPx: number;
  heightWorld: number;
  label: ClusterLabel;
  lines: string[];
  widthPx: number;
  widthWorld: number;
  x: number;
  y: number;
};

type TerritoryCell = {
  polygon: ScreenPoint[];
  region: FilmAtlasTerritoryRegion;
};

type TerritoryCellsByLayer = Record<FilmAtlasLayer, TerritoryCell[]>;
type GmapCell = Omit<FilmAtlasGmapCell, "polygon"> & {
  bounds: Bounds;
  polygon: ScreenPoint[];
};
type GmapBoundaryEdge = {
  cells: GmapCell[];
  from: ScreenPoint;
  key: string;
  to: ScreenPoint;
};
type GmapBoundarySegment = {
  from: ScreenPoint;
  to: ScreenPoint;
};
type GmapBoundaryEdgesByLayer = Record<FilmAtlasLayer, GmapBoundaryEdge[]>;
type OrganicEdge = {
  from: ScreenPoint;
  isShared: boolean;
  key: string;
  regions: FilmAtlasTerritoryRegion[];
  samples: ScreenPoint[];
  to: ScreenPoint;
};

type OrganicEdgeMapByLayer = Record<FilmAtlasLayer, Map<string, OrganicEdge>>;
type AtlasColorMode = "macro" | "micro" | "neighborhood";
type TerritoryRenderMode = "biological" | "coastal" | "dense_coast" | "gmap" | "organic" | "territory";

type TerritoryRenderSpec = {
  description: string;
  edgeAmplitudeScale: number;
  fillAlphaScale: number;
  frame: "coast" | "superellipse";
  frameMargin: number;
  frameWave: number;
  label: string;
  organicEdges: boolean;
  outerStrokeScale: number;
  pointFillScale: number;
  pointSampleScale: number;
  regionRadiusScale: number;
  strokeAlphaScale: number;
  viewportPaddingRatio: number;
};

type AtlasLayoutMode = {
  description: string;
  gmapCells: GmapCell[];
  id: string;
  isTerritory: boolean;
  label: string;
  pointsById: Map<number, ScreenPoint>;
  regions: FilmAtlasTerritoryRegion[];
};

type Bounds = {
  maxX: number;
  maxY: number;
  minX: number;
  minY: number;
};

type ScreenPoint = {
  x: number;
  y: number;
};

type ScreenRect = {
  bottom: number;
  left: number;
  right: number;
  top: number;
};

type HslColor = {
  hue: number;
  lightness: number;
  saturation: number;
};

type InitialFilmAtlasExport = Omit<FilmAtlasExport, "neighbors">;

const FILES: Record<keyof InitialFilmAtlasExport, string> = {
  labels: "labels.json",
  macro_clusters: "macro_clusters.json",
  manifest: "manifest.json",
  micro_clusters: "micro_clusters.json",
  movies: "movies.json",
  neighborhood_clusters: "neighborhood_clusters.json",
  points: "points.json",
};

const PALETTE = [
  "#f2c46d",
  "#e96d53",
  "#68d8c5",
  "#c184f4",
  "#f29bb2",
  "#8bdc78",
  "#77a8f7",
  "#f0a35f",
  "#c7d86d",
  "#6fd2f2",
  "#f26f8f",
  "#d9c7a3",
];

const COLOR_MODE_LABELS: Partial<Record<AtlasColorMode, string>> = {
  neighborhood: "Neighborhood shades",
};

const TERRITORY_LAYER_SETTINGS: Record<FilmAtlasLayer, {
  fillAlpha: number;
  inflate: number;
  strokeAlpha: number;
}> = {
  macro: {
    fillAlpha: 0.075,
    inflate: 1,
    strokeAlpha: 0.36,
  },
  neighborhood: {
    fillAlpha: 0.018,
    inflate: 1,
    strokeAlpha: 0.18,
  },
  micro: {
    fillAlpha: 0.004,
    inflate: 1,
    strokeAlpha: 0.075,
  },
};

const TERRITORY_RENDER_ORDER: TerritoryRenderMode[] = ["gmap"];

const TERRITORY_RENDER_SPECS: Partial<Record<TerritoryRenderMode, TerritoryRenderSpec>> = {
  gmap: {
    description: "Cell borders keep each film in its semantic position while revealing the active atlas tier.",
    edgeAmplitudeScale: 1,
    fillAlphaScale: 1.16,
    frame: "superellipse",
    frameMargin: 0.025,
    frameWave: 0,
    label: "Cell borders",
    organicEdges: false,
    outerStrokeScale: 1,
    pointFillScale: 1,
    pointSampleScale: 1,
    regionRadiusScale: 1,
    strokeAlphaScale: 1.18,
    viewportPaddingRatio: 0.048,
  },
};

const MIN_ZOOM = 0.72;
const MAX_ZOOM = 16;
const LABEL_TRANSITION_MS = 280;
const EDGE_POINT_PRECISION = 10000;

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

const distance = (a: ScreenPoint, b: ScreenPoint) => Math.hypot(a.x - b.x, a.y - b.y);

const hashString = (value: string) => {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
};

const organicPointKey = (point: ScreenPoint) =>
  `${Math.round(point.x * EDGE_POINT_PRECISION)},${Math.round(point.y * EDGE_POINT_PRECISION)}`;

const organicEdgeKey = (from: ScreenPoint, to: ScreenPoint) => {
  const fromKey = organicPointKey(from);
  const toKey = organicPointKey(to);
  return fromKey <= toKey ? `${fromKey}|${toKey}` : `${toKey}|${fromKey}`;
};

const midpoint = (a: ScreenPoint, b: ScreenPoint): ScreenPoint => ({
  x: (a.x + b.x) / 2,
  y: (a.y + b.y) / 2,
});

const macroPaletteColor = (macroId: number | null | undefined) =>
  PALETTE[Math.abs(macroId ?? 0) % PALETTE.length];

const hexToHsl = (hex: string): HslColor => {
  const normalized = hex.replace("#", "");
  const red = parseInt(normalized.slice(0, 2), 16) / 255;
  const green = parseInt(normalized.slice(2, 4), 16) / 255;
  const blue = parseInt(normalized.slice(4, 6), 16) / 255;
  const max = Math.max(red, green, blue);
  const min = Math.min(red, green, blue);
  const lightness = (max + min) / 2;
  const delta = max - min;

  if (delta === 0) {
    return { hue: 0, lightness: lightness * 100, saturation: 0 };
  }

  const saturation = delta / (1 - Math.abs(2 * lightness - 1));
  let hue = 0;
  if (max === red) {
    hue = ((green - blue) / delta) % 6;
  } else if (max === green) {
    hue = (blue - red) / delta + 2;
  } else {
    hue = (red - green) / delta + 4;
  }

  return {
    hue: (hue * 60 + 360) % 360,
    lightness: lightness * 100,
    saturation: saturation * 100,
  };
};

const hslToCss = (color: HslColor) =>
  `hsl(${Math.round((color.hue + 360) % 360)}, ${Math.round(clamp(color.saturation, 0, 100))}%, ${Math.round(clamp(color.lightness, 0, 100))}%)`;

const neighborhoodShadeHsl = (
  macroId: number | null | undefined,
  neighborhoodId: number | null | undefined,
) => {
  const base = hexToHsl(macroPaletteColor(macroId));
  if (neighborhoodId === null || neighborhoodId === undefined) return base;
  const seed = Math.abs(neighborhoodId);
  const hueOffset = (((seed * 29) % 21) - 10) * 1.45;
  const saturationOffset = (((seed * 17) % 9) - 4) * 3.3;
  const lightnessSteps = [-14, -9, -4, 2, 7, 12, 17];
  const lightnessOffset = lightnessSteps[seed % lightnessSteps.length] ?? 0;
  return {
    hue: (base.hue + hueOffset + 360) % 360,
    lightness: clamp(base.lightness + lightnessOffset, 52, 78),
    saturation: clamp(base.saturation + saturationOffset, 58, 94),
  };
};

const neighborhoodShadeColor = (
  macroId: number | null | undefined,
  neighborhoodId: number | null | undefined,
) => hslToCss(neighborhoodShadeHsl(macroId, neighborhoodId));

const microShadeColor = (
  macroId: number | null | undefined,
  neighborhoodId: number | null | undefined,
  microId: number | null | undefined,
) => {
  if (microId === null || microId === undefined) {
    return neighborhoodShadeColor(macroId, neighborhoodId);
  }

  const base = neighborhoodShadeHsl(macroId, neighborhoodId);
  const seed = Math.abs(microId * 37 + (neighborhoodId ?? 0) * 11);
  const hueSteps = [-26, -20, -14, -8, -3, 4, 10, 16, 23, 30];
  const lightnessSteps = [-18, -13, -8, -3, 3, 8, 13, 18];
  const hueOffset = hueSteps[seed % hueSteps.length] ?? 0;
  const saturationOffset = (((seed * 19) % 11) - 5) * 2.4;
  const lightnessOffset = lightnessSteps[Math.floor(seed / 3) % lightnessSteps.length] ?? 0;

  return hslToCss({
    hue: (base.hue + hueOffset + 360) % 360,
    lightness: clamp(base.lightness + lightnessOffset, 48, 82),
    saturation: clamp(base.saturation + saturationOffset, 60, 96),
  });
};

const emptyTerritoryCells = (): TerritoryCellsByLayer => ({
  macro: [],
  micro: [],
  neighborhood: [],
});

const emptyGmapBoundaryEdges = (): GmapBoundaryEdgesByLayer => ({
  macro: [],
  micro: [],
  neighborhood: [],
});

const emptyGmapCellIndex = (): Record<FilmAtlasLayer, Map<number, GmapCell[]>> => ({
  macro: new Map(),
  micro: new Map(),
  neighborhood: new Map(),
});

const emptyPointPositionMaps = (): Record<FilmAtlasLayer, Map<number, ScreenPoint>> => ({
  macro: new Map(),
  micro: new Map(),
  neighborhood: new Map(),
});

const emptyOrganicEdgeMaps = (): OrganicEdgeMapByLayer => ({
  macro: new Map(),
  micro: new Map(),
  neighborhood: new Map(),
});

const labelTextForLayer = (layer: FilmAtlasLayer) => {
  if (layer === "macro") return "Macro clusters";
  if (layer === "neighborhood") return "Neighborhoods";
  return "Microclusters";
};

const normalizeText = (value: string | number | null | undefined) =>
  String(value ?? "")
    .toLocaleLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "");

const renderModeSpec = (mode: TerritoryRenderMode) => TERRITORY_RENDER_SPECS[mode] ?? TERRITORY_RENDER_SPECS.gmap!;

const renderModeLabel = (mode: TerritoryRenderMode) => renderModeSpec(mode).label;

const layoutModeLabel = (mode: AtlasLayoutMode) => {
  const labels: Record<string, string> = {
    semantic_gmap_cells: "Semantic Cells",
  };
  return labels[mode.id] ?? mode.label;
};

const parseTerritoryRenderMode = (value: string): TerritoryRenderMode =>
  TERRITORY_RENDER_ORDER.includes(value as TerritoryRenderMode)
    ? value as TerritoryRenderMode
    : "gmap";

const parseColorMode = (value: string): AtlasColorMode =>
  value === "neighborhood" ? value : "neighborhood";

const formatYear = (movie: FilmAtlasMovie) => {
  if (typeof movie.year === "number") return String(movie.year);
  if (movie.release_date) return movie.release_date.slice(0, 4);
  return "Unknown year";
};

const formatRuntime = (runtime: number | null | undefined) => {
  if (!runtime) return "";
  const hours = Math.floor(runtime / 60);
  const minutes = runtime % 60;
  if (!hours) return `${minutes} min`;
  return `${hours}h ${minutes}m`;
};

const labelFor = (
  layer: FilmAtlasLayer,
  id: number | null,
  labelsByKey: Map<string, string>,
  fallback = "Unlabeled",
) => {
  if (id === null) return fallback;
  return labelsByKey.get(`${layer}:${id}`) ?? fallback;
};

const clusterKey = (layer: FilmAtlasLayer, id: number | null) =>
  id === null ? null : `${layer}:${id}`;

const getPointClusterId = (node: AtlasNode, layer: FilmAtlasLayer) => {
  if (layer === "macro") return node.macroId;
  if (layer === "neighborhood") return node.neighborhoodId;
  return node.microId;
};

const getPointLayerLabel = (node: AtlasNode, layer: FilmAtlasLayer) => {
  if (layer === "macro") return node.macroLabel;
  if (layer === "neighborhood") return node.neighborhoodLabel;
  return node.microLabel;
};

const getElement = <T extends Element>(root: ParentNode, selector: string) => {
  const element = root.querySelector<T>(selector);
  if (!element) {
    throw new Error(`Film Atlas element missing: ${selector}`);
  }
  return element;
};

const getOptionalElement = <T extends Element>(root: ParentNode, selector: string) =>
  root.querySelector<T>(selector);

async function fetchJson<T>(baseUrl: string, fileName: string): Promise<T> {
  const response = await fetch(`${baseUrl.replace(/\/$/, "")}/${fileName}`);

  if (!response.ok) {
    throw new Error(`Could not load ${fileName}: ${response.status}`);
  }

  return response.json() as Promise<T>;
}

async function loadAtlasData(baseUrl: string): Promise<InitialFilmAtlasExport> {
  const entries = await Promise.all(
    (Object.entries(FILES) as Array<[keyof InitialFilmAtlasExport, string]>).map(
      async ([key, file]) => [key, await fetchJson<InitialFilmAtlasExport[typeof key]>(baseUrl, file)] as const,
    ),
  );

  return Object.fromEntries(entries) as InitialFilmAtlasExport;
}

async function loadTerritoryLayouts(baseUrl: string): Promise<FilmAtlasTerritoryLayouts | null> {
  const response = await fetch(`${baseUrl.replace(/\/$/, "")}/territory_layouts.json`, {
    cache: "no-cache",
  });
  if (response.status === 404) return null;
  if (!response.ok) {
    console.warn(`Could not load territory layouts: ${response.status}`);
    return null;
  }
  return response.json() as Promise<FilmAtlasTerritoryLayouts>;
}

function buildLabelMap(labels: FilmAtlasLabel[], clusterSets: Array<[FilmAtlasLayer, FilmAtlasCluster[]]>) {
  const labelsByKey = new Map<string, string>();

  for (const label of labels) {
    const text = label.recommended_label || label.plain_label;
    if (text) labelsByKey.set(`${label.layer}:${label.cluster_id}`, text);
  }

  for (const [layer, clusters] of clusterSets) {
    for (const cluster of clusters) {
      if (cluster.recommended_label) {
        labelsByKey.set(`${layer}:${cluster.cluster_id}`, cluster.recommended_label);
      }
    }
  }

  return labelsByKey;
}

function buildNodes(data: InitialFilmAtlasExport) {
  const moviesById = new Map(data.movies.map((movie) => [movie.tmdb_id, movie]));
  const labelsByKey = buildLabelMap(data.labels, [
    ["macro", data.macro_clusters],
    ["neighborhood", data.neighborhood_clusters],
    ["micro", data.micro_clusters],
  ]);

  return data.points
    .map((point): AtlasNode | null => {
      const movie = moviesById.get(point.tmdb_id);
      if (!movie) return null;

      const macroId = typeof point.macro_id === "number" ? point.macro_id : null;
      const neighborhoodId = typeof point.neighborhood_id === "number" ? point.neighborhood_id : null;
      const microId = typeof point.micro_id === "number" ? point.micro_id : null;
      const macroLabel = labelFor("macro", macroId, labelsByKey);
      const neighborhoodLabel = labelFor("neighborhood", neighborhoodId, labelsByKey, macroLabel);
      const microLabel = labelFor("micro", microId, labelsByKey, neighborhoodLabel);
      const title = movie.title || movie.original_title || `Movie ${movie.tmdb_id}`;
      const year = formatYear(movie);

      return {
        color: PALETTE[Math.abs(macroId ?? 0) % PALETTE.length],
        macroId,
        macroLabel,
        mainLabel: macroLabel,
        microId,
        microLabel,
        movie: { ...movie, title },
        neighborhoodId,
        neighborhoodLabel,
        point,
        searchText: normalizeText([
          title,
          movie.original_title,
          year,
          macroLabel,
          neighborhoodLabel,
          microLabel,
          ...(movie.genres ?? []),
        ].join(" ")),
      };
    })
    .filter((node): node is AtlasNode => node !== null);
}

function getBounds(
  nodes: AtlasNode[],
  coordinateForNode: (node: AtlasNode) => ScreenPoint,
  regions: FilmAtlasTerritoryRegion[] = [],
  gmapCells: GmapCell[] = [],
): Bounds {
  const pointBounds = nodes.reduce(
    (bounds, node) => ({
      maxX: Math.max(bounds.maxX, coordinateForNode(node).x),
      maxY: Math.max(bounds.maxY, coordinateForNode(node).y),
      minX: Math.min(bounds.minX, coordinateForNode(node).x),
      minY: Math.min(bounds.minY, coordinateForNode(node).y),
    }),
    { maxX: -Infinity, maxY: -Infinity, minX: Infinity, minY: Infinity },
  );

  const regionBounds = regions.reduce(
    (bounds, region) => ({
      maxX: Math.max(bounds.maxX, region.x + region.radius),
      maxY: Math.max(bounds.maxY, region.y + region.radius),
      minX: Math.min(bounds.minX, region.x - region.radius),
      minY: Math.min(bounds.minY, region.y - region.radius),
    }),
    pointBounds,
  );
  return gmapCells.reduce(
    (bounds, cell) => ({
      maxX: Math.max(bounds.maxX, cell.bounds.maxX),
      maxY: Math.max(bounds.maxY, cell.bounds.maxY),
      minX: Math.min(bounds.minX, cell.bounds.minX),
      minY: Math.min(bounds.minY, cell.bounds.minY),
    }),
    regionBounds,
  );
}

function boundsForPoints(points: ScreenPoint[]): Bounds {
  return points.reduce(
    (bounds, point) => ({
      maxX: Math.max(bounds.maxX, point.x),
      maxY: Math.max(bounds.maxY, point.y),
      minX: Math.min(bounds.minX, point.x),
      minY: Math.min(bounds.minY, point.y),
    }),
    { maxX: -Infinity, maxY: -Infinity, minX: Infinity, minY: Infinity },
  );
}

function buildClusterLabels(
  nodes: AtlasNode[],
  layer: FilmAtlasLayer,
  coordinateForNode: (node: AtlasNode) => ScreenPoint,
): ClusterLabel[] {
  const groups = new Map<string, {
    color: string;
    count: number;
    id: number;
    label: string;
    x: number;
    y: number;
  }>();

  for (const node of nodes) {
    const id = getPointClusterId(node, layer);
    const key = clusterKey(layer, id);
    if (key === null || id === null) continue;

    const current = groups.get(key) ?? {
      color: node.color,
      count: 0,
      id,
      label: getPointLayerLabel(node, layer),
      x: 0,
      y: 0,
    };

    current.count += 1;
    const coordinate = coordinateForNode(node);
    current.x += coordinate.x;
    current.y += coordinate.y;
    groups.set(key, current);
  }

  return Array.from(groups.values())
    .map((group) => ({
      ...group,
      layer,
      x: group.x / group.count,
      y: group.y / group.count,
    }))
    .sort((a, b) => b.count - a.count);
}

function getLabelLayer(scale: number): FilmAtlasLayer {
  if (scale < 3.15) return "macro";
  if (scale < 7.1) return "neighborhood";
  return "micro";
}

function getLabelLines(text: string, layer: FilmAtlasLayer) {
  const maxLineLength = layer === "macro" ? 23 : layer === "neighborhood" ? 24 : 24;
  const clean = text
    .replace(/\s+/g, " ")
    .replace(/\s+&\s+/g, " & ")
    .trim();
  const words = clean.split(" ");
  const lines: string[] = [];
  let current = "";

  for (const word of words) {
    const next = current ? `${current} ${word}` : word;
    if (next.length <= maxLineLength || !current) {
      current = next;
      continue;
    }

    lines.push(current);
    current = word;
  }

  if (current) lines.push(current);

  return lines.length > 0 ? lines : [clean];
}

function scoreSearch(node: AtlasNode, query: string) {
  const title = normalizeText(node.movie.title);
  const original = normalizeText(node.movie.original_title);
  const year = formatYear(node.movie);

  if (title === query || original === query) return 0;
  if (title.startsWith(query) || original.startsWith(query)) return 1;
  const titleIndex = title.indexOf(query);
  if (titleIndex >= 0) return 2 + titleIndex / 100;
  if (year.startsWith(query)) return 3;
  const searchIndex = node.searchText.indexOf(query);
  if (searchIndex >= 0) return 4 + searchIndex / 1000;
  return Infinity;
}

export function initFilmAtlas(root: HTMLElement) {
  const dataBase = root.dataset.dataBase;
  if (!dataBase) return;

  const canvas = getElement<HTMLCanvasElement>(root, "[data-atlas-canvas]");
  const stage = getElement<HTMLElement>(root, "[data-atlas-stage]");
  const tooltip = getElement<HTMLElement>(root, "[data-atlas-tooltip]");
  const status = getElement<HTMLElement>(root, "[data-atlas-status]");
  const labelTier = getElement<HTMLElement>(root, "[data-atlas-label-tier]");
  const searchInput = getElement<HTMLInputElement>(root, "[data-atlas-search]");
  const searchResults = getElement<HTMLElement>(root, "[data-atlas-search-results]");
  const searchForm = getElement<HTMLFormElement>(root, "[data-atlas-search-form]");
  const resetButton = getElement<HTMLButtonElement>(root, "[data-atlas-reset]");
  const zoomInButton = getElement<HTMLButtonElement>(root, "[data-atlas-zoom-in]");
  const zoomOutButton = getElement<HTMLButtonElement>(root, "[data-atlas-zoom-out]");
  const layoutSelect = getOptionalElement<HTMLSelectElement>(root, "[data-atlas-layout-select]");
  const renderSelect = getOptionalElement<HTMLSelectElement>(root, "[data-atlas-render-select]");
  const colorSelect = getOptionalElement<HTMLSelectElement>(root, "[data-atlas-color-select]");
  const selectedTitle = getElement<HTMLElement>(root, "[data-atlas-selected-title]");
  const selectedMeta = getElement<HTMLElement>(root, "[data-atlas-selected-meta]");
  const selectedOverview = getElement<HTMLElement>(root, "[data-atlas-selected-overview]");
  const selectedLabels = getElement<HTMLElement>(root, "[data-atlas-selected-labels]");
  const selectedGenres = getElement<HTMLElement>(root, "[data-atlas-selected-genres]");
  const selectedChip = getElement<HTMLButtonElement>(root, "[data-atlas-selected-chip]");
  const selectedChipTitle = getElement<HTMLElement>(root, "[data-atlas-selected-chip-title]");
  const neighborSection = getElement<HTMLElement>(root, "[data-atlas-neighbor-section]");
  const neighborList = getElement<HTMLElement>(root, "[data-atlas-neighbors]");
  const movieCount = getElement<HTMLElement>(root, "[data-atlas-movie-count]");
  const clusterCount = getElement<HTMLElement>(root, "[data-atlas-cluster-count]");
  const generatedAt = getElement<HTMLElement>(root, "[data-atlas-generated-at]");

  const ctx = canvas.getContext("2d", { alpha: true });
  if (!ctx) {
    status.textContent = "Canvas is unavailable in this browser.";
    return;
  }

  let nodes: AtlasNode[] = [];
  let labelsByLayer: Record<FilmAtlasLayer, ClusterLabel[]> = {
    macro: [],
    micro: [],
    neighborhood: [],
  };
  let gmapPointPositions = new Map<number, ScreenPoint>();
  let gmapCellsByCluster = emptyGmapCellIndex();
  let gmapBoundaryEdgesByLayer: GmapBoundaryEdgesByLayer = emptyGmapBoundaryEdges();
  let territoryCells: TerritoryCellsByLayer = emptyTerritoryCells();
  let organicEdgesByLayer: OrganicEdgeMapByLayer = emptyOrganicEdgeMaps();
  let territoryPointPositions: Record<FilmAtlasLayer, Map<number, ScreenPoint>> = emptyPointPositionMaps();
  let territoryRenderMode: TerritoryRenderMode = "gmap";
  let layoutModes: AtlasLayoutMode[] = [];
  let activeLayout: AtlasLayoutMode | null = null;
  let activeColorMode: AtlasColorMode = "neighborhood";
  let neighborsById = new Map<number, FilmAtlasNeighborRecord>();
  let neighborShardCount = 100;
  let neighborShardDirectory = "neighbor_shards";
  const loadedNeighborShards = new Set<number>();
  const loadingNeighborShards = new Map<number, Promise<void>>();
  let nodesById = new Map<number, AtlasNode>();
  let bounds: Bounds = { maxX: 1, maxY: 1, minX: 0, minY: 0 };
  let cssWidth = 1;
  let cssHeight = 1;
  let dpr = 1;
  let baseScale = 1;
  let zoom = 1;
  let offsetX = 0;
  let offsetY = 0;
  let hovered: AtlasNode | null = null;
  let selected: AtlasNode | null = null;
  let isDragging = false;
  let hasDragged = false;
  let dragStartX = 0;
  let dragStartY = 0;
  let dragOffsetX = 0;
  let dragOffsetY = 0;
  let gestureStartDistance = 0;
  let gestureStartZoom = 1;
  let gestureAnchorWorld: ScreenPoint | null = null;
  const activePointers = new Map<number, ScreenPoint>();
  let activeLabelLayer: FilmAtlasLayer = "macro";
  let previousLabelLayer: FilmAtlasLayer | null = null;
  let labelTransitionStartedAt = 0;
  let cameraInitialized = false;
  let pendingDraw = 0;
  let currentSearchResults: AtlasNode[] = [];
  let activeSearchIndex = -1;

  const projectionCoordinateForNode = (node: AtlasNode): ScreenPoint => ({
    x: node.point.x,
    y: node.point.y,
  });

  const originalCoordinateForNode = (node: AtlasNode): ScreenPoint =>
    activeLayout?.pointsById.get(node.movie.tmdb_id) ?? projectionCoordinateForNode(node);

  const currentRenderSpec = () => renderModeSpec(territoryRenderMode);

  const usesTerritoryPointFill = () =>
    activeLayout?.isTerritory && territoryRenderMode !== "biological" && territoryRenderMode !== "gmap";

  const coordinateForNode = (node: AtlasNode): ScreenPoint => {
    if (territoryRenderMode === "gmap") {
      return gmapPointPositions.get(node.movie.tmdb_id)
        ?? originalCoordinateForNode(node);
    }
    if (usesTerritoryPointFill()) {
      return territoryPointPositions[activeLabelLayer].get(node.movie.tmdb_id)
        ?? originalCoordinateForNode(node);
    }
    return originalCoordinateForNode(node);
  };

  const centerX = () => (bounds.minX + bounds.maxX) / 2;
  const centerY = () => (bounds.minY + bounds.maxY) / 2;

  const worldToScreen = (x: number, y: number): ScreenPoint => ({
    x: (x - centerX()) * baseScale * zoom + cssWidth / 2 + offsetX,
    y: (centerY() - y) * baseScale * zoom + cssHeight / 2 + offsetY,
  });

  const screenToWorld = (x: number, y: number): ScreenPoint => ({
    x: (x - cssWidth / 2 - offsetX) / (baseScale * zoom) + centerX(),
    y: centerY() - (y - cssHeight / 2 - offsetY) / (baseScale * zoom),
  });

  const setStatus = (message: string, visible = true) => {
    status.textContent = message;
    status.hidden = !visible;
    status.style.display = visible ? "" : "none";
  };

  const requestDraw = () => {
    if (pendingDraw) return;
    pendingDraw = window.requestAnimationFrame(() => {
      pendingDraw = 0;
      draw();
    });
  };

  const updateLabelTransition = () => {
    const nextLayer = getLabelLayer(zoom);
    if (nextLayer !== activeLabelLayer) {
      previousLabelLayer = activeLabelLayer;
      activeLabelLayer = nextLayer;
      labelTransitionStartedAt = performance.now();
      labelTier.textContent = labelTextForLayer(nextLayer);
      if (selected && territoryRenderMode !== "biological") {
        const coordinate = coordinateForNode(selected);
        moveTooltip(selected, worldToScreen(coordinate.x, coordinate.y));
      }
    }
  };

  const labelTransitionProgress = () => {
    if (!previousLabelLayer) return 1;
    return clamp((performance.now() - labelTransitionStartedAt) / LABEL_TRANSITION_MS, 0, 1);
  };

  const fitToView = () => {
    const width = Math.max(1, bounds.maxX - bounds.minX);
    const height = Math.max(1, bounds.maxY - bounds.minY);
    const padding = activeLayout?.isTerritory
      ? Math.max(18, Math.min(cssWidth, cssHeight) * currentRenderSpec().viewportPaddingRatio)
      : Math.max(34, Math.min(cssWidth, cssHeight) * 0.1);
    baseScale = Math.min((cssWidth - padding * 2) / width, (cssHeight - padding * 2) / height);
    if (!Number.isFinite(baseScale) || baseScale <= 0) baseScale = 1;

    if (!cameraInitialized) {
      zoom = 1;
      offsetX = 0;
      offsetY = 0;
      cameraInitialized = true;
    }
  };

  const buildLayoutMode = (variant: FilmAtlasTerritoryVariant): AtlasLayoutMode => ({
    description: variant.description || "Nested territory experiment.",
    gmapCells: (variant.gmap_cells ?? []).map((cell) => {
      const polygon = cell.polygon.map(([x, y]) => ({ x, y }));
      return {
        bounds: boundsForPoints(polygon),
        macro_id: cell.macro_id,
        micro_id: cell.micro_id,
        neighborhood_id: cell.neighborhood_id,
        polygon,
        tmdb_id: cell.tmdb_id,
      };
    }),
    id: variant.id,
    isTerritory: true,
    label: variant.label,
    pointsById: new Map(variant.points.map((point) => [
      point.tmdb_id,
      { x: point.x, y: point.y },
    ])),
    regions: variant.regions,
  });

  const territoryWeightScale = (layer: FilmAtlasLayer) => {
    if (layer === "macro") return 0.7;
    if (layer === "neighborhood") return 0.56;
    return 0.48;
  };

  const smoothClosedPolygon = (points: ScreenPoint[], iterations: number) => {
    let smoothed = points;
    for (let iteration = 0; iteration < iterations; iteration += 1) {
      const nextPoints: ScreenPoint[] = [];
      for (let index = 0; index < smoothed.length; index += 1) {
        const current = smoothed[index];
        const next = smoothed[(index + 1) % smoothed.length];
        if (!current || !next) continue;
        nextPoints.push({
          x: current.x * 0.75 + next.x * 0.25,
          y: current.y * 0.75 + next.y * 0.25,
        });
        nextPoints.push({
          x: current.x * 0.25 + next.x * 0.75,
          y: current.y * 0.25 + next.y * 0.75,
        });
      }
      smoothed = nextPoints;
    }
    return smoothed;
  };

  const superellipseFramePolygon = (spec: TerritoryRenderSpec) => {
    const width = Math.max(1, bounds.maxX - bounds.minX);
    const height = Math.max(1, bounds.maxY - bounds.minY);
    const span = Math.max(width, height);
    const radiusX = width / 2 + span * spec.frameMargin;
    const radiusY = height / 2 + span * spec.frameMargin;
    const exponent = 2.75;
    const points: ScreenPoint[] = [];

    for (let index = 0; index < 72; index += 1) {
      const angle = (index / 72) * Math.PI * 2;
      const cos = Math.cos(angle);
      const sin = Math.sin(angle);
      points.push({
        x: centerX() + Math.sign(cos) * Math.abs(cos) ** (2 / exponent) * radiusX,
        y: centerY() + Math.sign(sin) * Math.abs(sin) ** (2 / exponent) * radiusY,
      });
    }

    return points;
  };

  const coastalFramePolygon = (spec: TerritoryRenderSpec) => {
    const macroRegions = activeLayout?.regions.filter((region) => region.layer === "macro") ?? [];
    if (macroRegions.length < 3) return superellipseFramePolygon(spec);

    const origin = {
      x: macroRegions.reduce((sum, region) => sum + region.x * (region.size ?? 1), 0)
        / Math.max(1, macroRegions.reduce((sum, region) => sum + (region.size ?? 1), 0)),
      y: macroRegions.reduce((sum, region) => sum + region.y * (region.size ?? 1), 0)
        / Math.max(1, macroRegions.reduce((sum, region) => sum + (region.size ?? 1), 0)),
    };
    const span = Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY, 1);
    const points: ScreenPoint[] = [];
    const steps = territoryRenderMode === "dense_coast" ? 112 : 96;

    for (let index = 0; index < steps; index += 1) {
      const angle = (index / steps) * Math.PI * 2;
      const ux = Math.cos(angle);
      const uy = Math.sin(angle);
      let maxDistance = -Infinity;

      for (const region of macroRegions) {
        const dx = region.x - origin.x;
        const dy = region.y - origin.y;
        const projection = dx * ux + dy * uy;
        const cross = dx * -uy + dy * ux;
        const radius = region.radius * spec.regionRadiusScale;
        const discriminant = radius ** 2 - cross ** 2;
        if (discriminant < 0) continue;
        maxDistance = Math.max(maxDistance, projection + Math.sqrt(discriminant));
      }

      if (!Number.isFinite(maxDistance)) {
        maxDistance = span * 0.52;
      }

      const wave = Math.sin(angle * 3 + 0.7) * 0.52
        + Math.sin(angle * 7 - 0.25) * 0.31
        + Math.cos(angle * 11 + 1.1) * 0.17;
      const minDistance = maxDistance + span * 0.004;
      const coastDistance = Math.max(
        minDistance,
        maxDistance + span * spec.frameMargin + span * spec.frameWave * wave,
      );
      points.push({
        x: origin.x + ux * coastDistance,
        y: origin.y + uy * coastDistance,
      });
    }

    return smoothClosedPolygon(points, 1);
  };

  const atlasFramePolygon = () => {
    const spec = currentRenderSpec();
    return spec.frame === "coast"
      ? coastalFramePolygon(spec)
      : superellipseFramePolygon(spec);
  };

  const clipPolygonToPowerCell = (
    polygon: ScreenPoint[],
    region: FilmAtlasTerritoryRegion,
    sibling: FilmAtlasTerritoryRegion,
    layer: FilmAtlasLayer,
  ) => {
    if (polygon.length < 3) return [];

    const scale = territoryWeightScale(layer);
    const regionWeight = region.radius * scale;
    const siblingWeight = sibling.radius * scale;
    const a = 2 * (sibling.x - region.x);
    const b = 2 * (sibling.y - region.y);
    const c = sibling.x ** 2 + sibling.y ** 2
      - region.x ** 2 - region.y ** 2
      + regionWeight ** 2 - siblingWeight ** 2;
    const signedDistance = (point: ScreenPoint) => a * point.x + b * point.y - c;
    const intersection = (from: ScreenPoint, to: ScreenPoint): ScreenPoint => {
      const fromDistance = signedDistance(from);
      const toDistance = signedDistance(to);
      const ratio = Math.abs(fromDistance - toDistance) < 1e-9
        ? 0.5
        : clamp(fromDistance / (fromDistance - toDistance), 0, 1);
      return {
        x: from.x + (to.x - from.x) * ratio,
        y: from.y + (to.y - from.y) * ratio,
      };
    };
    const clipped: ScreenPoint[] = [];
    let previous = polygon[polygon.length - 1];
    if (!previous) return [];
    let previousInside = signedDistance(previous) <= 1e-7;

    for (const current of polygon) {
      const currentInside = signedDistance(current) <= 1e-7;
      if (currentInside !== previousInside) {
        clipped.push(intersection(previous, current));
      }
      if (currentInside) clipped.push(current);
      previous = current;
      previousInside = currentInside;
    }

    return clipped;
  };

  const buildCellsForLayer = (
    layer: FilmAtlasLayer,
    rootPolygon: ScreenPoint[],
    parentPolygons: Map<number, ScreenPoint[]>,
  ) => {
    const layerRegions = activeLayout?.regions.filter((region) => region.layer === layer) ?? [];
    const groups = new Map<string, FilmAtlasTerritoryRegion[]>();
    for (const region of layerRegions) {
      const key = layer === "macro" ? "root" : String(region.parent_cluster_id ?? "root");
      groups.set(key, [...(groups.get(key) ?? []), region]);
    }

    const cells: TerritoryCell[] = [];
    const polygonsById = new Map<number, ScreenPoint[]>();
    for (const regions of groups.values()) {
      for (const region of regions) {
        const parentPolygon = layer === "macro"
          ? rootPolygon
          : parentPolygons.get(region.parent_cluster_id ?? -1);
        if (!parentPolygon?.length) continue;

        let polygon = parentPolygon.map((point) => ({ ...point }));
        for (const sibling of regions) {
          if (sibling.cluster_id === region.cluster_id) continue;
          polygon = clipPolygonToPowerCell(polygon, region, sibling, layer);
          if (polygon.length < 3) break;
        }

        if (polygon.length >= 3) {
          cells.push({ polygon, region });
          polygonsById.set(region.cluster_id, polygon);
        }
      }
    }

    return { cells, polygonsById };
  };

  const rebuildTerritoryCells = () => {
    territoryCells = emptyTerritoryCells();
    if (!activeLayout?.isTerritory) return;

    const rootPolygon = atlasFramePolygon();
    const macro = buildCellsForLayer("macro", rootPolygon, new Map());
    const neighborhood = buildCellsForLayer("neighborhood", rootPolygon, macro.polygonsById);
    const micro = buildCellsForLayer("micro", rootPolygon, neighborhood.polygonsById);

    territoryCells = {
      macro: macro.cells,
      micro: micro.cells,
      neighborhood: neighborhood.cells,
    };
  };

  const gmapClusterId = (cell: GmapCell, layer: FilmAtlasLayer) => {
    if (layer === "macro") return cell.macro_id;
    if (layer === "neighborhood") return cell.neighborhood_id;
    return cell.micro_id;
  };

  const rebuildGmapCellIndex = () => {
    gmapCellsByCluster = emptyGmapCellIndex();
    if (!activeLayout?.gmapCells.length) return;

    for (const cell of activeLayout.gmapCells) {
      for (const layer of ["macro", "neighborhood", "micro"] as FilmAtlasLayer[]) {
        const clusterId = gmapClusterId(cell, layer);
        const cells = gmapCellsByCluster[layer].get(clusterId) ?? [];
        cells.push(cell);
        gmapCellsByCluster[layer].set(clusterId, cells);
      }
    }
  };

  const rebuildGmapBoundaryEdges = () => {
    gmapBoundaryEdgesByLayer = emptyGmapBoundaryEdges();
    if (!activeLayout?.gmapCells.length) return;

    const edgeRefs = new Map<string, {
      cells: GmapCell[];
      from: ScreenPoint;
      to: ScreenPoint;
    }>();
    for (const cell of activeLayout.gmapCells) {
      for (let index = 0; index < cell.polygon.length; index += 1) {
        const from = cell.polygon[index];
        const to = cell.polygon[(index + 1) % cell.polygon.length];
        if (!from || !to) continue;
        const key = organicEdgeKey(from, to);
        const existing = edgeRefs.get(key);
        if (existing) {
          existing.cells.push(cell);
        } else {
          edgeRefs.set(key, { cells: [cell], from, to });
        }
      }
    }

    for (const layer of ["macro", "neighborhood", "micro"] as FilmAtlasLayer[]) {
      const edges = gmapBoundaryEdgesByLayer[layer];
      for (const [key, edge] of edgeRefs) {
        const firstCell = edge.cells[0];
        const secondCell = edge.cells[1];
        if (!firstCell) continue;
        if (!secondCell || gmapClusterId(firstCell, layer) !== gmapClusterId(secondCell, layer)) {
          edges.push({ cells: edge.cells, from: edge.from, key, to: edge.to });
        }
      }
    }
  };

  const rebuildGmapPointPositions = () => {
    gmapPointPositions = new Map();
    if (!activeLayout?.isTerritory || !activeLayout.gmapCells.length) return;

    const groupKeyForIds = (
      macroId: number | null | undefined,
      neighborhoodId: number | null | undefined,
      microId: number | null | undefined,
    ) => `${macroId ?? "x"}:${neighborhoodId ?? "x"}:${microId ?? "x"}`;

    const cellsByMicro = new Map<string, GmapCell[]>();
    for (const cell of activeLayout.gmapCells) {
      const key = groupKeyForIds(cell.macro_id, cell.neighborhood_id, cell.micro_id);
      cellsByMicro.set(key, [...(cellsByMicro.get(key) ?? []), cell]);
    }

    const nodesByMicro = new Map<string, AtlasNode[]>();
    for (const node of nodes) {
      const key = groupKeyForIds(node.macroId, node.neighborhoodId, node.microId);
      nodesByMicro.set(key, [...(nodesByMicro.get(key) ?? []), node]);
    }

    for (const [key, clusterNodes] of nodesByMicro) {
      const cells = cellsByMicro.get(key);
      if (!cells?.length) continue;
      const center = gmapCellUnionCenter(cells);
      const sortedNodes = [...clusterNodes].sort((a, b) => {
        const aCoordinate = originalCoordinateForNode(a);
        const bCoordinate = originalCoordinateForNode(b);
        const angleA = Math.atan2(aCoordinate.y - center.y, aCoordinate.x - center.x);
        const angleB = Math.atan2(bCoordinate.y - center.y, bCoordinate.x - center.x);
        return angleA - angleB || a.movie.tmdb_id - b.movie.tmdb_id;
      });
      const samples = sampleGmapCellUnion(cells, sortedNodes.length, hashString(key));
      for (const [index, node] of sortedNodes.entries()) {
        const sample = samples[index];
        if (sample) gmapPointPositions.set(node.movie.tmdb_id, sample);
      }
    }
  };

  const organicEdgeAmplitude = (layer: FilmAtlasLayer, length: number, isShared: boolean) => {
    const span = Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY, 1);
    const spanLimit = layer === "macro" ? 0.0062 : layer === "neighborhood" ? 0.0037 : 0.0019;
    const lengthLimit = layer === "macro" ? 0.048 : layer === "neighborhood" ? 0.037 : 0.028;
    const sharedScale = isShared ? 1 : 0.34;
    return Math.min(length * lengthLimit, span * spanLimit) * sharedScale * currentRenderSpec().edgeAmplitudeScale;
  };

  const buildOrganicEdge = (
    key: string,
    from: ScreenPoint,
    to: ScreenPoint,
    layer: FilmAtlasLayer,
    isShared: boolean,
    regions: FilmAtlasTerritoryRegion[],
  ): OrganicEdge => {
    const fromKey = organicPointKey(from);
    const toKey = organicPointKey(to);
    const start = fromKey <= toKey ? from : to;
    const end = fromKey <= toKey ? to : from;
    const length = distance(start, end);
    if (length < 1e-9) {
      return { from: start, isShared, key, regions, samples: [start, end], to: end };
    }

    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const normal = { x: -dy / length, y: dx / length };
    const seed = hashString(key);
    const phase = ((seed % 10000) / 10000) * Math.PI * 2;
    const direction = seed % 2 === 0 ? 1 : -1;
    const amplitude = organicEdgeAmplitude(layer, length, isShared);
    const baseSegments = layer === "macro" ? 11 : layer === "neighborhood" ? 8 : 6;
    const segments = Math.max(4, Math.min(14, Math.round(baseSegments + Math.sqrt(length) * 0.35)));
    const samples: ScreenPoint[] = [];

    for (let index = 0; index <= segments; index += 1) {
      const t = index / segments;
      const taper = Math.sin(Math.PI * t);
      const primaryWave = Math.sin(Math.PI * 2 * t + phase) * 0.62;
      const secondaryWave = Math.sin(Math.PI * 4 * t + phase * 1.37) * 0.24;
      const tertiaryWave = Math.cos(Math.PI * 3 * t + phase * 0.73) * 0.14;
      const offset = (primaryWave + secondaryWave + tertiaryWave) * taper * amplitude * direction;
      samples.push({
        x: start.x + dx * t + normal.x * offset,
        y: start.y + dy * t + normal.y * offset,
      });
    }

    return { from: start, isShared, key, regions, samples, to: end };
  };

  const rebuildOrganicEdges = () => {
    organicEdgesByLayer = emptyOrganicEdgeMaps();
    if (!activeLayout?.isTerritory) return;

    for (const layer of ["macro", "neighborhood", "micro"] as FilmAtlasLayer[]) {
      const edgeCounts = new Map<string, number>();
      const edgeRefs = new Map<string, {
        from: ScreenPoint;
        regions: FilmAtlasTerritoryRegion[];
        to: ScreenPoint;
      }>();

      for (const cell of territoryCells[layer]) {
        for (let index = 0; index < cell.polygon.length; index += 1) {
          const from = cell.polygon[index];
          const to = cell.polygon[(index + 1) % cell.polygon.length];
          if (!from || !to) continue;
          const key = organicEdgeKey(from, to);
          edgeCounts.set(key, (edgeCounts.get(key) ?? 0) + 1);
          const existing = edgeRefs.get(key);
          if (existing) {
            existing.regions.push(cell.region);
          } else {
            edgeRefs.set(key, { from, regions: [cell.region], to });
          }
        }
      }

      const edgeMap = organicEdgesByLayer[layer];
      for (const [key, reference] of edgeRefs) {
        edgeMap.set(
          key,
          buildOrganicEdge(
            key,
            reference.from,
            reference.to,
            layer,
            (edgeCounts.get(key) ?? 0) > 1,
            reference.regions,
          ),
        );
      }
    }
  };

  const polygonBounds = (polygon: ScreenPoint[]) => polygon.reduce((acc, point) => ({
    maxX: Math.max(acc.maxX, point.x),
    maxY: Math.max(acc.maxY, point.y),
    minX: Math.min(acc.minX, point.x),
    minY: Math.min(acc.minY, point.y),
  }), { maxX: -Infinity, maxY: -Infinity, minX: Infinity, minY: Infinity });

  const polygonCentroid = (polygon: ScreenPoint[]) => {
    if (polygon.length === 0) return { x: 0, y: 0 };
    let twiceArea = 0;
    let x = 0;
    let y = 0;
    for (let index = 0; index < polygon.length; index += 1) {
      const current = polygon[index];
      const next = polygon[(index + 1) % polygon.length];
      if (!current || !next) continue;
      const cross = current.x * next.y - next.x * current.y;
      twiceArea += cross;
      x += (current.x + next.x) * cross;
      y += (current.y + next.y) * cross;
    }
    if (Math.abs(twiceArea) < 1e-9) {
      return {
        x: polygon.reduce((sum, point) => sum + point.x, 0) / polygon.length,
        y: polygon.reduce((sum, point) => sum + point.y, 0) / polygon.length,
      };
    }
    return {
      x: x / (3 * twiceArea),
      y: y / (3 * twiceArea),
    };
  };

  const pointInPolygon = (point: ScreenPoint, polygon: ScreenPoint[]) => {
    let inside = false;
    for (let index = 0, previousIndex = polygon.length - 1; index < polygon.length; previousIndex = index, index += 1) {
      const current = polygon[index];
      const previous = polygon[previousIndex];
      if (!current || !previous) continue;
      const intersects = ((current.y > point.y) !== (previous.y > point.y))
        && point.x < (previous.x - current.x) * (point.y - current.y) / (previous.y - current.y + 1e-12) + current.x;
      if (intersects) inside = !inside;
    }
    return inside;
  };

  const halton = (index: number, base: number) => {
    let result = 0;
    let fraction = 1 / base;
    let value = index;
    while (value > 0) {
      result += fraction * (value % base);
      value = Math.floor(value / base);
      fraction /= base;
    }
    return result;
  };

  const polygonArea = (polygon: ScreenPoint[]) => {
    let twiceArea = 0;
    for (let index = 0; index < polygon.length; index += 1) {
      const current = polygon[index];
      const next = polygon[(index + 1) % polygon.length];
      if (!current || !next) continue;
      twiceArea += current.x * next.y - next.x * current.y;
    }
    return Math.abs(twiceArea) / 2;
  };

  const gmapCellInfos = (cells: GmapCell[]) =>
    cells
      .filter((cell) => cell.polygon.length >= 3)
      .map((cell) => ({
        bounds: polygonBounds(cell.polygon),
        centroid: polygonCentroid(cell.polygon),
        polygon: cell.polygon,
        weight: Math.max(polygonArea(cell.polygon), 1e-8),
      }));

  const gmapCellUnionCenter = (cells: GmapCell[]): ScreenPoint => {
    const infos = gmapCellInfos(cells);
    if (!infos.length) return { x: 0, y: 0 };
    const totals = infos.reduce(
      (acc, info) => ({
        weight: acc.weight + info.weight,
        x: acc.x + info.centroid.x * info.weight,
        y: acc.y + info.centroid.y * info.weight,
      }),
      { weight: 0, x: 0, y: 0 },
    );
    return {
      x: totals.x / Math.max(totals.weight, 1e-8),
      y: totals.y / Math.max(totals.weight, 1e-8),
    };
  };

  const pointInGmapCellInfo = (
    point: ScreenPoint,
    info: ReturnType<typeof gmapCellInfos>[number],
  ) => point.x >= info.bounds.minX
    && point.x <= info.bounds.maxX
    && point.y >= info.bounds.minY
    && point.y <= info.bounds.maxY
    && pointInPolygon(point, info.polygon);

  const pointToSegmentDistance = (point: ScreenPoint, segment: GmapBoundarySegment) => {
    const dx = segment.to.x - segment.from.x;
    const dy = segment.to.y - segment.from.y;
    const lengthSquared = dx * dx + dy * dy;
    if (lengthSquared < 1e-12) return distance(point, segment.from);
    const projection = clamp(
      ((point.x - segment.from.x) * dx + (point.y - segment.from.y) * dy) / lengthSquared,
      0,
      1,
    );
    return distance(point, {
      x: segment.from.x + dx * projection,
      y: segment.from.y + dy * projection,
    });
  };

  const gmapUnionBoundarySegments = (cells: GmapCell[]) => {
    const edges = new Map<string, GmapBoundarySegment & { count: number }>();
    for (const cell of cells) {
      for (let index = 0; index < cell.polygon.length; index += 1) {
        const from = cell.polygon[index];
        const to = cell.polygon[(index + 1) % cell.polygon.length];
        if (!from || !to) continue;
        const key = organicEdgeKey(from, to);
        const existing = edges.get(key);
        if (existing) {
          existing.count += 1;
        } else {
          edges.set(key, { count: 1, from, to });
        }
      }
    }
    return [...edges.values()]
      .filter((edge) => edge.count === 1)
      .map(({ from, to }) => ({ from, to }));
  };

  const gmapUnionBoundaryPadding = (bounds: Bounds) => {
    const span = Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY, 1e-8);
    return clamp(span * 0.028, 0.18, 0.72);
  };

  const pointHasBoundaryPadding = (
    point: ScreenPoint,
    boundarySegments: GmapBoundarySegment[],
    padding: number,
  ) => !boundarySegments.length
    || boundarySegments.every((segment) => pointToSegmentDistance(point, segment) >= padding);

  const sampleGmapCellUnion = (cells: GmapCell[], count: number, seed: number) => {
    if (count <= 0) return [];
    const infos = gmapCellInfos(cells);
    if (!infos.length) return [];

    const bounds = infos.reduce(
      (acc, info) => ({
        maxX: Math.max(acc.maxX, info.bounds.maxX),
        maxY: Math.max(acc.maxY, info.bounds.maxY),
        minX: Math.min(acc.minX, info.bounds.minX),
        minY: Math.min(acc.minY, info.bounds.minY),
      }),
      { maxX: -Infinity, maxY: -Infinity, minX: Infinity, minY: Infinity },
    );
    const center = gmapCellUnionCenter(cells);
    const boundarySegments = gmapUnionBoundarySegments(cells);
    const boundaryPadding = gmapUnionBoundaryPadding(bounds);
    const samples: ScreenPoint[] = [];
    const maxAttempts = Math.max(360, count * 240);
    const seedA = seed % 997;
    const seedB = seed % 991;

    const collectSamples = (minPadding: number, attemptOffset: number) => {
      for (let attempt = 1; attempt <= maxAttempts && samples.length < count; attempt += 1) {
        const sample = {
          x: bounds.minX + halton(attempt + attemptOffset + seedA, 2) * (bounds.maxX - bounds.minX),
          y: bounds.minY + halton(attempt + attemptOffset + seedB, 3) * (bounds.maxY - bounds.minY),
        };
        if (
          infos.some((info) => pointInGmapCellInfo(sample, info))
          && pointHasBoundaryPadding(sample, boundarySegments, minPadding)
        ) {
          samples.push(sample);
        }
      }
    };

    collectSamples(boundaryPadding, 0);
    if (samples.length < count) {
      collectSamples(boundaryPadding * 0.5, maxAttempts);
    }

    for (let attempt = 1; attempt <= maxAttempts && samples.length < count; attempt += 1) {
      const sample = {
        x: bounds.minX + halton(attempt + maxAttempts * 2 + seedA, 2) * (bounds.maxX - bounds.minX),
        y: bounds.minY + halton(attempt + maxAttempts * 2 + seedB, 3) * (bounds.maxY - bounds.minY),
      };
      if (infos.some((info) => pointInGmapCellInfo(sample, info))) {
        samples.push(sample);
      }
    }

    if (samples.length < count) {
      for (const info of infos.sort((a, b) => a.centroid.x - b.centroid.x || a.centroid.y - b.centroid.y)) {
        if (samples.length >= count) break;
        samples.push(info.centroid);
      }
    }

    return samples
      .sort((a, b) => {
        const angleA = Math.atan2(a.y - center.y, a.x - center.x);
        const angleB = Math.atan2(b.y - center.y, b.x - center.x);
        return angleA - angleB || distance(a, center) - distance(b, center);
      })
      .slice(0, count);
  };

  const pointFillBlend = (layer: FilmAtlasLayer) => {
    const scale = currentRenderSpec().pointFillScale;
    if (layer === "macro") return clamp(0.88 * scale, 0.72, 0.96);
    if (layer === "neighborhood") return clamp(0.8 * scale, 0.64, 0.93);
    return clamp(0.68 * scale, 0.54, 0.88);
  };

  const buildCellSamplePoints = (
    cell: TerritoryCell,
    count: number,
    layer: FilmAtlasLayer,
  ) => {
    if (count <= 0) return [];
    const boundsForCell = polygonBounds(cell.polygon);
    const centroid = polygonCentroid(cell.polygon);
    const samples: ScreenPoint[] = [];
    const maxAttempts = Math.max(160, count * 90);
    const baseShrink = layer === "macro" ? 0.94 : layer === "neighborhood" ? 0.92 : 0.88;
    const shrink = clamp(baseShrink * currentRenderSpec().pointSampleScale, 0.78, 0.985);

    for (let attempt = 1; attempt <= maxAttempts && samples.length < count; attempt += 1) {
      const x = boundsForCell.minX + halton(attempt + cell.region.cluster_id * 11, 2) * (boundsForCell.maxX - boundsForCell.minX);
      const y = boundsForCell.minY + halton(attempt + cell.region.cluster_id * 17, 3) * (boundsForCell.maxY - boundsForCell.minY);
      const sample = {
        x: centroid.x + (x - centroid.x) * shrink,
        y: centroid.y + (y - centroid.y) * shrink,
      };
      if (pointInPolygon(sample, cell.polygon)) {
        samples.push(sample);
      }
    }

    if (samples.length < count) {
      for (let index = samples.length; index < count; index += 1) {
        const angle = index * Math.PI * (3 - Math.sqrt(5));
        const radius = Math.sqrt((index + 0.5) / count) * Math.min(
          boundsForCell.maxX - boundsForCell.minX,
          boundsForCell.maxY - boundsForCell.minY,
        ) * 0.36;
        samples.push({
          x: centroid.x + Math.cos(angle) * radius,
          y: centroid.y + Math.sin(angle) * radius,
        });
      }
    }

    return samples
      .sort((a, b) => {
        const angleA = Math.atan2(a.y - centroid.y, a.x - centroid.x);
        const angleB = Math.atan2(b.y - centroid.y, b.x - centroid.x);
        return angleA - angleB || distance(a, centroid) - distance(b, centroid);
      })
      .slice(0, count);
  };

  const rebuildTerritoryPointPositions = () => {
    territoryPointPositions = emptyPointPositionMaps();
    if (!activeLayout?.isTerritory || territoryRenderMode === "gmap") return;

    for (const layer of ["macro", "neighborhood", "micro"] as FilmAtlasLayer[]) {
      const cellsById = new Map(territoryCells[layer].map((cell) => [cell.region.cluster_id, cell]));
      const nodesByCluster = new Map<number, AtlasNode[]>();
      for (const node of nodes) {
        const clusterId = getPointClusterId(node, layer);
        if (clusterId === null || !cellsById.has(clusterId)) continue;
        nodesByCluster.set(clusterId, [...(nodesByCluster.get(clusterId) ?? []), node]);
      }

      for (const [clusterId, clusterNodes] of nodesByCluster) {
        const cell = cellsById.get(clusterId);
        if (!cell) continue;
        const centroid = polygonCentroid(cell.polygon);
        const sortedNodes = [...clusterNodes].sort((a, b) => {
          const aCoordinate = originalCoordinateForNode(a);
          const bCoordinate = originalCoordinateForNode(b);
          const angleA = Math.atan2(aCoordinate.y - cell.region.y, aCoordinate.x - cell.region.x);
          const angleB = Math.atan2(bCoordinate.y - cell.region.y, bCoordinate.x - cell.region.x);
          return angleA - angleB || a.movie.tmdb_id - b.movie.tmdb_id;
        });
        const samples = buildCellSamplePoints(cell, sortedNodes.length, layer);
        const fillBlend = pointFillBlend(layer);
        for (const [index, node] of sortedNodes.entries()) {
          const sample = samples[index] ?? centroid;
          const original = originalCoordinateForNode(node);
          const blended = {
            x: original.x * (1 - fillBlend) + sample.x * fillBlend,
            y: original.y * (1 - fillBlend) + sample.y * fillBlend,
          };
          territoryPointPositions[layer].set(
            node.movie.tmdb_id,
            pointInPolygon(blended, cell.polygon) ? blended : sample,
          );
        }
      }
    }
  };

  const rebuildLabelsForActiveLayout = () => {
    const labelSets = {
      macro: buildClusterLabels(nodes, "macro", coordinateForNode),
      micro: buildClusterLabels(nodes, "micro", coordinateForNode),
      neighborhood: buildClusterLabels(nodes, "neighborhood", coordinateForNode),
    };

    if (!activeLayout?.isTerritory) {
      labelsByLayer = labelSets;
      return;
    }

    const regionLabels = (layer: FilmAtlasLayer) => {
      const labelsById = new Map(labelSets[layer].map((label) => [label.id, label]));
      return activeLayout.regions
        .filter((region) => region.layer === layer)
        .map((region): ClusterLabel | null => {
          const label = labelsById.get(region.cluster_id);
          if (!label) return null;
          return {
            ...label,
            count: region.size ?? label.count,
            x: region.x,
            y: region.y,
          };
        })
        .filter((label): label is ClusterLabel => label !== null)
        .sort((a, b) => b.count - a.count);
    };

    labelsByLayer = {
      macro: regionLabels("macro"),
      micro: regionLabels("micro"),
      neighborhood: regionLabels("neighborhood"),
    };
  };

  const populateLayoutOptions = () => {
    if (!layoutSelect) return;
    layoutSelect.innerHTML = "";
    for (const mode of layoutModes) {
      const option = document.createElement("option");
      option.value = mode.id;
      option.textContent = layoutModeLabel(mode);
      layoutSelect.append(option);
    }
    layoutSelect.disabled = layoutModes.length <= 1;
  };

  const populateRenderOptions = () => {
    if (!renderSelect) return;
    renderSelect.innerHTML = "";
    for (const mode of TERRITORY_RENDER_ORDER) {
      const option = document.createElement("option");
      option.value = mode;
      option.textContent = renderModeLabel(mode);
      option.disabled = mode === "gmap" && !(activeLayout?.gmapCells.length);
      renderSelect.append(option);
    }
    renderSelect.value = territoryRenderMode;
    renderSelect.disabled = !activeLayout?.isTerritory || TERRITORY_RENDER_ORDER.length <= 1;
  };

  const populateColorOptions = () => {
    if (!colorSelect) return;
    colorSelect.innerHTML = "";
    for (const [value, label] of Object.entries(COLOR_MODE_LABELS)) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      colorSelect.append(option);
    }
    colorSelect.value = activeColorMode;
    colorSelect.disabled = Object.keys(COLOR_MODE_LABELS).length <= 1;
  };

  const setColorMode = (mode: AtlasColorMode) => {
    activeColorMode = mode;
    if (colorSelect) colorSelect.value = mode;
    if (selected) {
      const coordinate = coordinateForNode(selected);
      moveTooltip(selected, worldToScreen(coordinate.x, coordinate.y));
    }
    requestDraw();
  };

  const updateLayoutDescription = () => {
    return;
  };

  const setTerritoryRenderMode = (mode: TerritoryRenderMode) => {
    if (mode === "gmap" && !activeLayout?.gmapCells.length) {
      mode = "territory";
    }
    territoryRenderMode = mode;
    if (renderSelect) {
      renderSelect.value = mode;
      renderSelect.disabled = !activeLayout?.isTerritory || TERRITORY_RENDER_ORDER.length <= 1;
    }
    if (activeLayout?.isTerritory) {
      rebuildTerritoryCells();
      rebuildGmapCellIndex();
      rebuildGmapBoundaryEdges();
      rebuildGmapPointPositions();
      rebuildOrganicEdges();
      rebuildTerritoryPointPositions();
      rebuildLabelsForActiveLayout();
      fitToView();
    }
    updateLayoutDescription();
    if (selected) {
      const coordinate = coordinateForNode(selected);
      moveTooltip(selected, worldToScreen(coordinate.x, coordinate.y));
    }
    requestDraw();
  };

  const setActiveLayout = (id: string, options: { resetCamera?: boolean } = {}) => {
    activeLayout = layoutModes.find((mode) => mode.id === id) ?? layoutModes[0] ?? null;
    if (!activeLayout) return;
    if (layoutSelect) layoutSelect.value = activeLayout.id;
    populateRenderOptions();
    updateLayoutDescription();
    if (territoryRenderMode === "gmap" && !activeLayout.gmapCells.length) {
      territoryRenderMode = "territory";
      populateRenderOptions();
      updateLayoutDescription();
    } else if (activeLayout.gmapCells.length && activeLayout.id.includes("gmap")) {
      territoryRenderMode = "gmap";
      populateRenderOptions();
      updateLayoutDescription();
    }
    bounds = getBounds(nodes, originalCoordinateForNode, activeLayout.regions, activeLayout.gmapCells);
    rebuildTerritoryCells();
    rebuildGmapCellIndex();
    rebuildGmapBoundaryEdges();
    rebuildGmapPointPositions();
    rebuildOrganicEdges();
    rebuildTerritoryPointPositions();
    rebuildLabelsForActiveLayout();

    if (options.resetCamera) {
      zoom = 1;
      offsetX = 0;
      offsetY = 0;
      cameraInitialized = false;
      hideTooltip();
    }

    fitToView();
    if (selected && !options.resetCamera) {
      const screen = worldToScreen(coordinateForNode(selected).x, coordinateForNode(selected).y);
      moveTooltip(selected, screen);
    }
    renderSelected();
    requestDraw();
  };

  const resizeCanvas = () => {
    const rect = stage.getBoundingClientRect();
    cssWidth = Math.max(1, rect.width);
    cssHeight = Math.max(1, rect.height);
    dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.round(cssWidth * dpr);
    canvas.height = Math.round(cssHeight * dpr);
    canvas.style.width = `${cssWidth}px`;
    canvas.style.height = `${cssHeight}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    fitToView();
    requestDraw();
  };

  const drawBackground = () => {
    ctx.fillStyle = "#050505";
    ctx.fillRect(0, 0, cssWidth, cssHeight);
  };

  const atlasColor = (
    macroId: number | null | undefined,
    neighborhoodId: number | null | undefined,
    microId?: number | null | undefined,
  ) => {
    if (activeColorMode === "micro") return microShadeColor(macroId, neighborhoodId, microId);
    if (activeColorMode === "neighborhood") return neighborhoodShadeColor(macroId, neighborhoodId);
    return macroPaletteColor(macroId);
  };

  const atlasLayerColor = (
    layer: FilmAtlasLayer,
    macroId: number | null | undefined,
    neighborhoodId: number | null | undefined,
    microId?: number | null | undefined,
  ) => {
    if (layer === "macro") return macroPaletteColor(macroId);
    if (layer === "neighborhood") {
      return activeColorMode === "macro"
        ? macroPaletteColor(macroId)
        : neighborhoodShadeColor(macroId, neighborhoodId);
    }
    return atlasColor(macroId, neighborhoodId, microId);
  };

  const nodeColor = (node: AtlasNode) => atlasColor(node.macroId, node.neighborhoodId, node.microId);

  const regionColor = (region: FilmAtlasTerritoryRegion) => {
    const macroId = region.layer === "macro" ? region.cluster_id : region.macro_id;
    const neighborhoodId = region.layer === "neighborhood" ? region.cluster_id : region.neighborhood_id;
    const microId = region.layer === "micro" ? region.cluster_id : null;
    return atlasLayerColor(region.layer, macroId, neighborhoodId, microId);
  };

  const territorySettings = (layer: FilmAtlasLayer) => TERRITORY_LAYER_SETTINGS[layer];

  const applyRenderSpecToSettings = (
    settings: typeof TERRITORY_LAYER_SETTINGS[FilmAtlasLayer],
  ) => {
    const spec = currentRenderSpec();
    return {
      ...settings,
      fillAlpha: settings.fillAlpha * spec.fillAlphaScale,
      strokeAlpha: settings.strokeAlpha * spec.strokeAlphaScale,
    };
  };

  const organicRadius = (region: FilmAtlasTerritoryRegion, angle: number) => {
    const seed = (region.cluster_id + 1) * 0.417 + (region.macro_id ?? 0) * 0.113;
    const layerScale = region.layer === "macro" ? 0.065 : region.layer === "neighborhood" ? 0.052 : 0.032;
    return 1
      + Math.sin(angle * 3 + seed) * layerScale
      + Math.cos(angle * 5 - seed * 1.7) * layerScale * 0.58
      + Math.sin(angle * 7 + seed * 2.3) * layerScale * 0.34;
  };

  const drawBiologicalRegion = (
    region: FilmAtlasTerritoryRegion,
    settings: typeof TERRITORY_LAYER_SETTINGS[FilmAtlasLayer],
  ) => {
    const screen = worldToScreen(region.x, region.y);
    const radius = region.radius * settings.inflate * baseScale * zoom;
    if (radius < 1.5) return;
    if (
      screen.x + radius < -30
      || screen.x - radius > cssWidth + 30
      || screen.y + radius < -30
      || screen.y - radius > cssHeight + 30
    ) {
      return;
    }

    const steps = region.layer === "macro" ? 58 : region.layer === "neighborhood" ? 42 : 24;
    ctx.beginPath();
    for (let index = 0; index <= steps; index += 1) {
      const angle = (index / steps) * Math.PI * 2;
      const warpedRadius = radius * organicRadius(region, angle);
      const x = screen.x + Math.cos(angle) * warpedRadius;
      const y = screen.y + Math.sin(angle) * warpedRadius;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fillStyle = regionColor(region);
    if (settings.fillAlpha > 0) {
      ctx.globalAlpha = settings.fillAlpha * 0.72;
      ctx.fill();
    }
    ctx.globalAlpha = settings.strokeAlpha * 0.72;
    ctx.lineWidth = region.layer === "macro" ? 1.6 : region.layer === "neighborhood" ? 1 : 0.7;
    ctx.strokeStyle = regionColor(region);
    ctx.stroke();
  };

  const paintTerritoryPath = (
    region: FilmAtlasTerritoryRegion,
    worldPoints: ScreenPoint[],
    settings: typeof TERRITORY_LAYER_SETTINGS[FilmAtlasLayer],
    options: { fill?: boolean; stroke?: boolean } = {},
  ) => {
    if (worldPoints.length < 3) return;
    const shouldFill = options.fill ?? true;
    const shouldStroke = options.stroke ?? true;

    const screenPoints = worldPoints.map((point) => worldToScreen(point.x, point.y));
    const minX = Math.min(...screenPoints.map((point) => point.x));
    const maxX = Math.max(...screenPoints.map((point) => point.x));
    const minY = Math.min(...screenPoints.map((point) => point.y));
    const maxY = Math.max(...screenPoints.map((point) => point.y));
    if (maxX < -30 || minX > cssWidth + 30 || maxY < -30 || minY > cssHeight + 30) {
      return;
    }

    ctx.beginPath();
    for (const [index, point] of screenPoints.entries()) {
      if (index === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    }
    ctx.closePath();
    ctx.fillStyle = regionColor(region);
    if (shouldFill && settings.fillAlpha > 0) {
      ctx.globalAlpha = settings.fillAlpha;
      ctx.fill();
    }
    if (shouldStroke && settings.strokeAlpha > 0) {
      ctx.globalAlpha = settings.strokeAlpha;
      ctx.lineWidth = region.layer === "macro" ? 1.15 : region.layer === "neighborhood" ? 0.78 : 0.46;
      ctx.lineJoin = "round";
      ctx.strokeStyle = regionColor(region);
      ctx.stroke();
    }
  };

  const drawTerritoryCell = (
    cell: TerritoryCell,
    settings: typeof TERRITORY_LAYER_SETTINGS[FilmAtlasLayer],
  ) => {
    paintTerritoryPath(cell.region, cell.polygon, settings);
  };

  const organicSamplesForCellEdge = (
    layer: FilmAtlasLayer,
    from: ScreenPoint,
    to: ScreenPoint,
  ) => {
    const edge = organicEdgesByLayer[layer].get(organicEdgeKey(from, to));
    if (!edge) return [from, to];

    const localFromKey = organicPointKey(from);
    const localToKey = organicPointKey(to);
    const edgeFromKey = organicPointKey(edge.from);
    const edgeToKey = organicPointKey(edge.to);
    const samples = localFromKey === edgeFromKey && localToKey === edgeToKey
      ? edge.samples
      : [...edge.samples].reverse();

    return samples;
  };

  const drawOrganicTerritoryCell = (
    cell: TerritoryCell,
    settings: typeof TERRITORY_LAYER_SETTINGS[FilmAtlasLayer],
  ) => {
    const path: ScreenPoint[] = [];

    for (let index = 0; index < cell.polygon.length; index += 1) {
      const from = cell.polygon[index];
      const to = cell.polygon[(index + 1) % cell.polygon.length];
      if (!from || !to) continue;
      const samples = organicSamplesForCellEdge(cell.region.layer, from, to);
      for (const [sampleIndex, sample] of samples.entries()) {
        if (path.length > 0 && sampleIndex === 0) continue;
        path.push(sample);
      }
    }

    paintTerritoryPath(cell.region, path.length >= 3 ? path : cell.polygon, settings, { stroke: false });
  };

  const drawOrganicTerritoryEdges = (
    layer: FilmAtlasLayer,
    settings: typeof TERRITORY_LAYER_SETTINGS[FilmAtlasLayer],
  ) => {
    const edges = organicEdgesByLayer[layer];
    if (!edges.size || settings.strokeAlpha <= 0) return;

    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    for (const edge of edges.values()) {
      // Child cells inherit the parent boundary, so only shared child edges are stroked.
      if (!edge.isShared && layer !== "macro") continue;
      const screenPoints = edge.samples.map((point) => worldToScreen(point.x, point.y));
      const minX = Math.min(...screenPoints.map((point) => point.x));
      const maxX = Math.max(...screenPoints.map((point) => point.x));
      const minY = Math.min(...screenPoints.map((point) => point.y));
      const maxY = Math.max(...screenPoints.map((point) => point.y));
      if (maxX < -30 || minX > cssWidth + 30 || maxY < -30 || minY > cssHeight + 30) {
        continue;
      }

      const edgeRegion = [...edge.regions].sort((left, right) => left.cluster_id - right.cluster_id)[0];
      if (!edgeRegion) continue;
      const isOuterCoast = !edge.isShared && layer === "macro";
      const baseLineWidth = layer === "macro" ? 1.15 : layer === "neighborhood" ? 0.78 : 0.46;
      ctx.lineWidth = baseLineWidth * (isOuterCoast ? currentRenderSpec().outerStrokeScale : 1);
      ctx.globalAlpha = clamp(
        settings.strokeAlpha * (edge.isShared ? 0.92 : 0.58 * currentRenderSpec().outerStrokeScale),
        0,
        0.9,
      );
      ctx.strokeStyle = regionColor(edgeRegion);
      ctx.beginPath();
      for (const [index, point] of screenPoints.entries()) {
        if (index === 0) ctx.moveTo(point.x, point.y);
        else ctx.lineTo(point.x, point.y);
      }
      ctx.stroke();
    }

    ctx.restore();
  };

  const gmapCellColor = (cell: GmapCell, layer: FilmAtlasLayer = activeLabelLayer) =>
    atlasLayerColor(layer, cell.macro_id, cell.neighborhood_id, cell.micro_id);

  const drawGmapCellFill = (cell: GmapCell, alpha: number) => {
    if (cell.polygon.length < 3) return;
    const firstBoundsCorner = worldToScreen(cell.bounds.minX, cell.bounds.minY);
    const secondBoundsCorner = worldToScreen(cell.bounds.maxX, cell.bounds.maxY);
    const minX = Math.min(firstBoundsCorner.x, secondBoundsCorner.x);
    const maxX = Math.max(firstBoundsCorner.x, secondBoundsCorner.x);
    const minY = Math.min(firstBoundsCorner.y, secondBoundsCorner.y);
    const maxY = Math.max(firstBoundsCorner.y, secondBoundsCorner.y);
    if (maxX < -24 || minX > cssWidth + 24 || maxY < -24 || minY > cssHeight + 24) return;

    ctx.beginPath();
    for (let index = 0; index < cell.polygon.length; index += 1) {
      const point = cell.polygon[index];
      const screenPoint = worldToScreen(point.x, point.y);
      if (index === 0) ctx.moveTo(screenPoint.x, screenPoint.y);
      else ctx.lineTo(screenPoint.x, screenPoint.y);
    }
    ctx.closePath();
    ctx.fillStyle = gmapCellColor(cell);
    ctx.globalAlpha = alpha;
    ctx.fill();
  };

  const drawGmapBoundaryLayer = (layer: FilmAtlasLayer, alpha: number, width: number) => {
    const edges = gmapBoundaryEdgesByLayer[layer];
    if (!edges.length) return;
    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = width;
    for (const edge of edges) {
      const from = worldToScreen(edge.from.x, edge.from.y);
      const to = worldToScreen(edge.to.x, edge.to.y);
      if (
        Math.max(from.x, to.x) < -24
        || Math.min(from.x, to.x) > cssWidth + 24
        || Math.max(from.y, to.y) < -24
        || Math.min(from.y, to.y) > cssHeight + 24
      ) {
        continue;
      }
      const cell = edge.cells[0];
      if (!cell) continue;
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = gmapCellColor(cell, layer);
      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.stroke();
    }
    ctx.restore();
  };

  const drawGmapRegions = () => {
    if (!activeLayout?.gmapCells.length) return false;
    ctx.save();
    const fillAlpha = activeLabelLayer === "macro" ? 0.118 : activeLabelLayer === "neighborhood" ? 0.092 : 0.07;
    for (const cell of activeLayout.gmapCells) {
      drawGmapCellFill(cell, fillAlpha * currentRenderSpec().fillAlphaScale);
    }

    drawGmapBoundaryLayer("macro", activeLabelLayer === "macro" ? 0.52 : 0.24, activeLabelLayer === "macro" ? 1.45 : 0.95);
    if (activeLabelLayer !== "macro") {
      drawGmapBoundaryLayer("neighborhood", activeLabelLayer === "neighborhood" ? 0.42 : 0.18, activeLabelLayer === "neighborhood" ? 0.92 : 0.62);
    }
    if (activeLabelLayer === "micro") {
      drawGmapBoundaryLayer("micro", 0.32, 0.44);
    }
    ctx.restore();
    return true;
  };

  const visibleTerritoryLayers = () => {
    if (activeLabelLayer === "macro") return ["macro"] as FilmAtlasLayer[];
    if (activeLabelLayer === "neighborhood") return ["macro", "neighborhood"] as FilmAtlasLayer[];
    return ["macro", "neighborhood", "micro"] as FilmAtlasLayer[];
  };

  const visibleTerritorySettings = (layer: FilmAtlasLayer) => {
    const settings = territorySettings(layer);
    if (activeLabelLayer === "macro") return applyRenderSpecToSettings(settings);
    if (layer === "macro") {
      return applyRenderSpecToSettings({
        ...settings,
        fillAlpha: settings.fillAlpha * 0.42,
        strokeAlpha: settings.strokeAlpha * 0.42,
      });
    }
    if (activeLabelLayer === "micro" && layer === "neighborhood") {
      return applyRenderSpecToSettings({
        ...settings,
        fillAlpha: 0,
        strokeAlpha: settings.strokeAlpha * 0.48,
      });
    }
    if (layer === "micro") {
      return applyRenderSpecToSettings({
        ...settings,
        fillAlpha: 0,
        strokeAlpha: settings.strokeAlpha * 0.9,
      });
    }
    return applyRenderSpecToSettings(settings);
  };

  const drawTerritoryRegions = () => {
    if (!activeLayout?.isTerritory) return;
    if (territoryRenderMode === "gmap" && drawGmapRegions()) return;
    ctx.save();
    if (currentRenderSpec().organicEdges) {
      const layers = visibleTerritoryLayers();
      for (const layer of layers) {
        const settings = visibleTerritorySettings(layer);
        for (const cell of territoryCells[layer]) {
          drawOrganicTerritoryCell(cell, settings);
        }
      }
      for (const layer of layers) {
        drawOrganicTerritoryEdges(layer, visibleTerritorySettings(layer));
      }
      ctx.restore();
      return;
    }

    for (const layer of visibleTerritoryLayers()) {
      const settings = visibleTerritorySettings(layer);
      if (territoryRenderMode === "biological") {
        for (const region of activeLayout.regions.filter((item) => item.layer === layer)) {
          drawBiologicalRegion(region, settings);
        }
      } else {
        for (const cell of territoryCells[layer]) {
          drawTerritoryCell(cell, settings);
        }
      }
    }
    ctx.restore();
  };

  const drawNeighborLines = () => {
    if (!selected) return;
    const record = neighborsById.get(selected.movie.tmdb_id);
    if (!record?.neighbors?.length) return;

    const selectedCoordinate = coordinateForNode(selected);
    const from = worldToScreen(selectedCoordinate.x, selectedCoordinate.y);
    ctx.save();
    ctx.lineWidth = 1;
    for (const neighbor of record.neighbors.slice(0, 8)) {
      const node = nodesById.get(neighbor.tmdb_id);
      if (!node) continue;
      const coordinate = coordinateForNode(node);
      const to = worldToScreen(coordinate.x, coordinate.y);
      ctx.globalAlpha = 0.18 + clamp((neighbor.similarity ?? 0.5) - 0.45, 0, 0.25);
      ctx.strokeStyle = nodeColor(selected);
      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.stroke();
    }
    ctx.restore();
  };

  const neighborShardForMovie = (tmdbId: number) =>
    Math.abs(tmdbId) % Math.max(1, neighborShardCount);

  const loadNeighborShard = (tmdbId: number) => {
    const shardId = neighborShardForMovie(tmdbId);
    if (loadedNeighborShards.has(shardId)) return Promise.resolve();
    const existing = loadingNeighborShards.get(shardId);
    if (existing) return existing;

    const shardName = `${neighborShardDirectory.replace(/\/$/, "")}/${String(shardId).padStart(2, "0")}.json`;
    const promise = fetchJson<FilmAtlasNeighborRecord[]>(dataBase, shardName)
      .then((records) => {
        for (const record of records) {
          neighborsById.set(record.tmdb_id, record);
        }
        loadedNeighborShards.add(shardId);
      })
      .catch((error: unknown) => {
        loadedNeighborShards.add(shardId);
        console.warn(`Could not load ${shardName}`, error);
      })
      .finally(() => {
        loadingNeighborShards.delete(shardId);
      });
    loadingNeighborShards.set(shardId, promise);
    return promise;
  };

  const drawPoints = () => {
    const isGmapRender = territoryRenderMode === "gmap";
    const baseRadius = isGmapRender
      ? clamp(1.38 + Math.log2(zoom + 1) * 0.3, 1.38, 2.7)
      : clamp(1.05 + Math.log2(zoom + 1) * 0.34, 1.05, 2.6);

    ctx.save();
    for (const node of nodes) {
      const coordinate = coordinateForNode(node);
      const screen = worldToScreen(coordinate.x, coordinate.y);
      if (screen.x < -20 || screen.x > cssWidth + 20 || screen.y < -20 || screen.y > cssHeight + 20) {
        continue;
      }

      const isSelected = selected?.movie.tmdb_id === node.movie.tmdb_id;
      const isHovered = hovered?.movie.tmdb_id === node.movie.tmdb_id;
      const radius = isSelected ? baseRadius + 3.9 : isHovered ? baseRadius + 2.4 : baseRadius;

      ctx.globalAlpha = isSelected || isHovered ? 0.82 : isGmapRender ? 0.32 : 0.46;
      ctx.fillStyle = "#030303";
      ctx.beginPath();
      ctx.arc(screen.x, screen.y, radius + (isGmapRender ? 0.86 : 1.35), 0, Math.PI * 2);
      ctx.fill();

      ctx.globalAlpha = isSelected || isHovered ? 1 : isGmapRender ? 0.98 : activeLayout?.isTerritory ? 0.93 : 0.78;
      ctx.fillStyle = nodeColor(node);
      ctx.beginPath();
      ctx.arc(screen.x, screen.y, radius, 0, Math.PI * 2);
      ctx.fill();

      if (!isSelected && !isHovered && zoom < 2.2) {
        ctx.globalAlpha = isGmapRender ? 0.3 : activeLayout?.isTerritory ? 0.24 : 0.16;
        ctx.fillStyle = "#fff8ea";
        ctx.beginPath();
        ctx.arc(screen.x - radius * 0.18, screen.y - radius * 0.18, Math.max(0.35, radius * 0.32), 0, Math.PI * 2);
        ctx.fill();
      }

      if (isSelected) {
        ctx.globalAlpha = 0.74;
        ctx.lineWidth = 5.5;
        ctx.strokeStyle = "#030303";
        ctx.beginPath();
        ctx.arc(screen.x, screen.y, radius + 6.4, 0, Math.PI * 2);
        ctx.stroke();

        ctx.globalAlpha = 0.96;
        ctx.lineWidth = 2.6;
        ctx.strokeStyle = "#f2c46d";
        ctx.beginPath();
        ctx.arc(screen.x, screen.y, radius + 6.4, 0, Math.PI * 2);
        ctx.stroke();

        ctx.globalAlpha = 0.72;
        ctx.lineWidth = 1.2;
        ctx.strokeStyle = "#fff8ea";
        ctx.beginPath();
        ctx.arc(screen.x, screen.y, radius + 2.8, 0, Math.PI * 2);
        ctx.stroke();
      } else if (isHovered) {
        ctx.globalAlpha = 0.26;
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = nodeColor(node);
        ctx.beginPath();
        ctx.arc(screen.x, screen.y, radius + 5, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
    ctx.restore();
  };

  const labelSeedPositionInWorld = (
    label: ClusterLabel,
    labelIndex: number,
    layer: FilmAtlasLayer,
  ): ScreenPoint => {
    if (!activeLayout?.isTerritory || shouldAnchorLabelsToCells(layer)) {
      return { x: label.x, y: label.y };
    }

    const span = Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY, 1);
    const dx = label.x - centerX();
    const dy = label.y - centerY();
    const fallbackAngle = (label.id * 0.61803398875 + labelIndex * 0.173) * Math.PI * 2;
    const baseAngle = Math.hypot(dx, dy) > 0.001 ? Math.atan2(dy, dx) : fallbackAngle;
    const angle = baseAngle + Math.sin((label.id + 1) * 1.913) * 0.18;
    const perpendicular = angle + Math.PI / 2;
    const distanceScale = layer === "macro" ? 0.043 : layer === "neighborhood" ? 0.012 : 0.0044;
    const jitterScale = layer === "macro" ? 0.01 : layer === "neighborhood" ? 0.004 : 0.0018;
    const distance = span * distanceScale * (0.72 + ((label.id + labelIndex) % 5) * 0.08);
    const jitter = span * jitterScale * ((((label.id * 37 + labelIndex * 11) % 9) - 4) / 4);

    return {
      x: label.x + Math.cos(angle) * distance + Math.cos(perpendicular) * jitter,
      y: label.y + Math.sin(angle) * distance + Math.sin(perpendicular) * jitter,
    };
  };

  const labelReferenceZoom = (layer: FilmAtlasLayer) => {
    if (layer === "macro") return 1.2;
    if (layer === "neighborhood") return 3.35;
    return 7.35;
  };

  const labelMaxOffset = (layer: FilmAtlasLayer) => {
    const span = Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY, 1);
    if (layer === "macro") return span * 0.078;
    if (layer === "neighborhood") return span * 0.029;
    return span * 0.011;
  };

  const clampLabelToRegion = (
    placement: ClusterLabelPlacement,
    label: ClusterLabel,
    maxOffset: number,
  ) => {
    const dx = placement.x - label.x;
    const dy = placement.y - label.y;
    const distanceFromRegion = Math.hypot(dx, dy);
    if (distanceFromRegion <= maxOffset || distanceFromRegion === 0) return;
    const scale = maxOffset / distanceFromRegion;
    placement.x = label.x + dx * scale;
    placement.y = label.y + dy * scale;
  };

  const labelLayoutFontSize = (layer: FilmAtlasLayer) => {
    if (layer === "macro") return activeLayout?.isTerritory ? 12.1 : 12.8;
    if (layer === "neighborhood") return 11.4;
    return 11.8;
  };

  const labelFontSize = (layer: FilmAtlasLayer) => {
    const base = labelLayoutFontSize(layer);
    if (layer !== "macro") return base;

    const zoomProgress = clamp((zoom - MIN_ZOOM) / (1 - MIN_ZOOM), 0, 1);
    const minSize = activeLayout?.isTerritory ? 10.7 : 11.2;
    return minSize + (base - minSize) * zoomProgress;
  };

  const labelLineHeight = (layer: FilmAtlasLayer, fontSize: number) =>
    fontSize * (layer === "macro" ? 1.08 : 1.12);

  const labelCollisionPadding = (layer: FilmAtlasLayer) => {
    if (layer === "macro") return 7;
    if (layer === "neighborhood") return zoom >= 5.7 ? 2 : 5;
    return zoom >= 12 ? 1 : 4;
  };

  const shouldCullLabels = (layer: FilmAtlasLayer) => {
    if (layer === "macro") return false;
    if (layer === "neighborhood") return zoom < 5.7;
    return zoom < 12;
  };

  function shouldAnchorLabelsToCells(layer: FilmAtlasLayer) {
    if (layer === "macro") return false;
    if (layer === "neighborhood") return zoom >= 5;
    return true;
  }

  const labelScreenRect = (
    screen: ScreenPoint,
    width: number,
    height: number,
    layer: FilmAtlasLayer,
  ): ScreenRect => {
    const padding = labelCollisionPadding(layer);
    return {
      bottom: screen.y + height / 2 + padding,
      left: screen.x - width / 2 - padding,
      right: screen.x + width / 2 + padding,
      top: screen.y - height / 2 - padding,
    };
  };

  const rectsOverlap = (left: ScreenRect, right: ScreenRect) =>
    left.left < right.right
    && left.right > right.left
    && left.top < right.bottom
    && left.bottom > right.top;

  const rectOverlapArea = (left: ScreenRect, right: ScreenRect) => {
    const width = Math.max(0, Math.min(left.right, right.right) - Math.max(left.left, right.left));
    const height = Math.max(0, Math.min(left.bottom, right.bottom) - Math.max(left.top, right.top));
    return width * height;
  };

  const labelWorldRect = (
    point: ScreenPoint,
    width: number,
    height: number,
    padding: number,
  ): ScreenRect => ({
    bottom: point.y + height / 2 + padding,
    left: point.x - width / 2 - padding,
    right: point.x + width / 2 + padding,
    top: point.y - height / 2 - padding,
  });

  const minDistanceToBoundary = (point: ScreenPoint, segments: GmapBoundarySegment[]) =>
    segments.length
      ? Math.min(...segments.map((segment) => pointToSegmentDistance(point, segment)))
      : Infinity;

  const gmapLabelAnchorCandidates = (label: ClusterLabel, layer: FilmAtlasLayer) => {
    if (territoryRenderMode !== "gmap" || !activeLayout?.gmapCells.length) return [];
    const cells = gmapCellsByCluster[layer].get(label.id);
    if (!cells?.length) return [];

    const infos = gmapCellInfos(cells);
    if (!infos.length) return [];

    const unionCenter = gmapCellUnionCenter(cells);
    const boundarySegments = gmapUnionBoundarySegments(cells);
    const boundsForCells = infos.reduce(
      (acc, info) => ({
        maxX: Math.max(acc.maxX, info.bounds.maxX),
        maxY: Math.max(acc.maxY, info.bounds.maxY),
        minX: Math.min(acc.minX, info.bounds.minX),
        minY: Math.min(acc.minY, info.bounds.minY),
      }),
      { maxX: -Infinity, maxY: -Infinity, minX: Infinity, minY: Infinity },
    );
    const span = Math.max(boundsForCells.maxX - boundsForCells.minX, boundsForCells.maxY - boundsForCells.minY, 1e-8);
    const candidates: ScreenPoint[] = [unionCenter];
    const seen = new Set<string>();
    const addCandidate = (point: ScreenPoint) => {
      if (!infos.some((info) => pointInGmapCellInfo(point, info))) return;
      const key = `${Math.round(point.x * 10000)},${Math.round(point.y * 10000)}`;
      if (seen.has(key)) return;
      seen.add(key);
      candidates.push(point);
    };

    for (const info of [...infos].sort((left, right) => right.weight - left.weight).slice(0, 8)) {
      addCandidate(info.centroid);
    }

    const sampleCount = layer === "micro" ? 120 : layer === "neighborhood" ? 82 : 42;
    const seed = hashString(`${layer}:${label.id}:${label.label}`);
    for (let index = 1; index <= sampleCount; index += 1) {
      addCandidate({
        x: boundsForCells.minX + halton(index + seed % 997, 2) * (boundsForCells.maxX - boundsForCells.minX),
        y: boundsForCells.minY + halton(index + seed % 991, 3) * (boundsForCells.maxY - boundsForCells.minY),
      });
    }

    return candidates
      .map((point) => ({
        boundaryClearance: minDistanceToBoundary(point, boundarySegments),
        centerDistance: distance(point, unionCenter) / span,
        point,
      }))
      .sort((left, right) =>
        right.boundaryClearance - left.boundaryClearance
        || left.centerDistance - right.centerDistance)
      .map((candidate) => candidate.point);
  };

  const applyGmapLabelAnchors = (placements: ClusterLabelPlacement[], layer: FilmAtlasLayer) => {
    if (!shouldAnchorLabelsToCells(layer) || territoryRenderMode !== "gmap") return placements;

    const placedRects: ScreenRect[] = [];
    const padding = layer === "micro" ? 0.0048 : 0.0072;
    const overlapPenalty = layer === "micro" ? 42 : 55;

    for (const placement of placements) {
      const candidates = gmapLabelAnchorCandidates(placement.label, layer);
      if (!candidates.length) continue;

      let bestPoint = candidates[0] ?? { x: placement.x, y: placement.y };
      let bestScore = Infinity;
      for (const point of candidates) {
        const rect = labelWorldRect(point, placement.widthWorld, placement.heightWorld, padding);
        const overlap = placedRects.reduce((total, placed) => total + rectOverlapArea(rect, placed), 0);
        const originPull = distance(point, { x: placement.label.x, y: placement.label.y });
        const score = overlap * overlapPenalty + originPull;
        if (score < bestScore) {
          bestScore = score;
          bestPoint = point;
        }
        if (overlap === 0 && originPull < Math.max(placement.widthWorld, placement.heightWorld) * 0.18) {
          break;
        }
      }

      placement.x = bestPoint.x;
      placement.y = bestPoint.y;
      placedRects.push(labelWorldRect(bestPoint, placement.widthWorld, placement.heightWorld, padding));
    }

    return placements;
  };

  const buildStableLabelPlacements = (
    layer: FilmAtlasLayer,
    lineHeight: number,
  ): ClusterLabelPlacement[] => {
    const referenceScale = Math.max(1, baseScale * labelReferenceZoom(layer));
    const placements = labelsByLayer[layer].map((label, labelIndex) => {
      const lines = getLabelLines(label.label, layer);
      const widthPx = Math.max(...lines.map((line) => ctx.measureText(line).width));
      const heightPx = lines.length * lineHeight;
      const seed = labelSeedPositionInWorld(label, labelIndex, layer);
      return {
        heightPx,
        heightWorld: (heightPx + 10) / referenceScale,
        label,
        lines,
        widthPx,
        widthWorld: (widthPx + 18) / referenceScale,
        x: seed.x,
        y: seed.y,
      };
    });

    const shouldRelax = !shouldAnchorLabelsToCells(layer);
    const maxOffset = labelMaxOffset(layer);
    const iterations = shouldRelax ? layer === "macro" ? 24 : layer === "neighborhood" ? 14 : 9 : 0;
    const strength = layer === "macro" ? 0.96 : layer === "neighborhood" ? 0.68 : 0.48;

    if (shouldRelax) {
      for (const placement of placements) {
        clampLabelToRegion(placement, placement.label, maxOffset);
      }
    }

    for (let iteration = 0; iteration < iterations; iteration += 1) {
      for (let leftIndex = 0; leftIndex < placements.length; leftIndex += 1) {
        const left = placements[leftIndex];
        if (!left) continue;
        for (let rightIndex = leftIndex + 1; rightIndex < placements.length; rightIndex += 1) {
          const right = placements[rightIndex];
          if (!right) continue;
          const dx = right.x - left.x;
          const dy = right.y - left.y;
          const overlapX = (left.widthWorld + right.widthWorld) / 2 - Math.abs(dx);
          const overlapY = (left.heightWorld + right.heightWorld) / 2 - Math.abs(dy);
          if (overlapX <= 0 || overlapY <= 0) continue;

          const fallbackAngle = (
            (left.label.id + 1) * 0.531
            + (right.label.id + 1) * 0.317
            + iteration * 0.071
          ) * Math.PI * 2;
          const resolveAlongX = overlapX < overlapY;
          const direction = resolveAlongX
            ? (dx === 0 ? Math.cos(fallbackAngle) || 1 : Math.sign(dx))
            : (dy === 0 ? Math.sin(fallbackAngle) || 1 : Math.sign(dy));
          const push = (resolveAlongX ? overlapX : overlapY) * 0.5 * strength;
          const totalCount = Math.max(1, left.label.count + right.label.count);
          const leftShare = right.label.count / totalCount;
          const rightShare = left.label.count / totalCount;

          if (resolveAlongX) {
            left.x -= direction * push * leftShare;
            right.x += direction * push * rightShare;
          } else {
            left.y -= direction * push * leftShare;
            right.y += direction * push * rightShare;
          }

          clampLabelToRegion(left, left.label, maxOffset);
          clampLabelToRegion(right, right.label, maxOffset);
        }
      }
    }

    return applyGmapLabelAnchors(placements, layer);
  };

  const drawLabelSet = (layer: FilmAtlasLayer, alpha: number) => {
    const layoutFontSize = labelLayoutFontSize(layer);
    const layoutLineHeight = labelLineHeight(layer, layoutFontSize);
    const fontSize = labelFontSize(layer);
    const lineHeight = labelLineHeight(layer, fontSize);
    ctx.save();
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.font = `${layoutFontSize}px Inter, ui-sans-serif, system-ui`;

    const placements = buildStableLabelPlacements(layer, layoutLineHeight)
      .sort((left, right) => right.label.count - left.label.count || left.label.id - right.label.id);

    ctx.font = `${fontSize}px Inter, ui-sans-serif, system-ui`;

    const drawnRects: ScreenRect[] = [];
    for (const placement of placements) {
      const label = placement.label;
      const screen = worldToScreen(placement.x, placement.y);
      const lines = placement.lines;
      const width = Math.max(...lines.map((line) => ctx.measureText(line).width));
      const height = lines.length * lineHeight;
      const viewportMargin = layer === "macro" ? 180 : layer === "neighborhood" ? 36 : 24;
      if (
        screen.x < -viewportMargin - width / 2
        || screen.x > cssWidth + viewportMargin + width / 2
        || screen.y < -viewportMargin - height / 2
        || screen.y > cssHeight + viewportMargin + height / 2
      ) {
        continue;
      }

      if (shouldCullLabels(layer)) {
        const rect = labelScreenRect(screen, width + 10, height + 6, layer);
        if (drawnRects.some((drawn) => rectsOverlap(rect, drawn))) continue;
        drawnRects.push(rect);
      }

      ctx.globalAlpha = alpha * (layer === "macro" ? 0.18 : layer === "neighborhood" ? 0.14 : 0.1);
      ctx.fillStyle = "#050505";
      ctx.fillRect(
        screen.x - width / 2 - 5,
        screen.y - height / 2 - 3,
        width + 10,
        height + 6,
      );

      ctx.lineWidth = layer === "macro" ? 7 : 5;
      ctx.strokeStyle = "rgba(5, 5, 5, 0.94)";
      ctx.globalAlpha = alpha * (layer === "macro" ? 0.98 : 0.72);
      ctx.fillStyle = layer === "macro" ? label.color : "#f7ead2";
      lines.forEach((line, lineIndex) => {
        const y = screen.y + (lineIndex - (lines.length - 1) / 2) * lineHeight;
        ctx.strokeText(line, screen.x, y);
        ctx.fillText(line, screen.x, y);
      });
      ctx.globalAlpha = 1;
    }
    ctx.restore();
  };

  const drawLabels = () => {
    const progress = labelTransitionProgress();
    if (previousLabelLayer) {
      drawLabelSet(previousLabelLayer, 1 - progress);
    }
    drawLabelSet(activeLabelLayer, progress);

    if (previousLabelLayer && progress < 1) {
      requestDraw();
    } else if (previousLabelLayer) {
      previousLabelLayer = null;
    }
  };

  function draw() {
    updateLabelTransition();
    drawBackground();
    drawTerritoryRegions();
    drawNeighborLines();
    drawPoints();
    drawLabels();
  }

  const findNearest = (x: number, y: number) => {
    let nearest: AtlasNode | null = null;
    let nearestDistance = Infinity;
    const hitRadius = clamp(10 + zoom * 1.5, 10, 18);

    for (const node of nodes) {
      const coordinate = coordinateForNode(node);
      const screen = worldToScreen(coordinate.x, coordinate.y);
      const distance = (screen.x - x) ** 2 + (screen.y - y) ** 2;
      if (distance < nearestDistance) {
        nearestDistance = distance;
        nearest = node;
      }
    }

    return nearestDistance <= hitRadius ** 2 ? nearest : null;
  };

  const moveTooltip = (node: AtlasNode, point: ScreenPoint) => {
    const title = node.movie.title || node.movie.original_title || "Untitled";
    tooltip.innerHTML = "";

    const titleElement = document.createElement("div");
    titleElement.className = "atlas-tooltip-title";
    titleElement.textContent = title;
    tooltip.append(titleElement);

    const metaElement = document.createElement("div");
    metaElement.className = "atlas-tooltip-meta";
    metaElement.textContent = `${formatYear(node.movie)} - ${node.mainLabel}`;
    tooltip.append(metaElement);

    tooltip.hidden = false;
    const rect = tooltip.getBoundingClientRect();
    const x = clamp(point.x + 14, 10, Math.max(10, cssWidth - rect.width - 10));
    const y = clamp(point.y + 14, 10, Math.max(10, cssHeight - rect.height - 10));
    tooltip.style.transform = `translate(${x}px, ${y}px)`;
  };

  const hideTooltip = () => {
    tooltip.hidden = true;
  };

  const clusterMembers = (layer: FilmAtlasLayer, id: number | null) => {
    if (id === null) return [];
    return nodes
      .filter((node) => getPointClusterId(node, layer) === id)
      .sort((a, b) => {
        if (selected?.movie.tmdb_id === a.movie.tmdb_id) return -1;
        if (selected?.movie.tmdb_id === b.movie.tmdb_id) return 1;
        return (a.movie.title || "").localeCompare(b.movie.title || "");
      });
  };

  const renderLabelPill = (name: string, value: string) => {
    const pill = document.createElement("div");
    pill.className = "atlas-label-pill";

    const key = document.createElement("span");
    key.className = "atlas-label-key";
    key.textContent = name;
    pill.append(key);

    const text = document.createElement("span");
    text.textContent = value;
    pill.append(text);
    return pill;
  };

  const renderClusterDisclosure = (
    name: string,
    value: string,
    layer: FilmAtlasLayer,
    clusterId: number | null,
  ) => {
    const members = clusterMembers(layer, clusterId);
    const details = document.createElement("details");
    details.className = "atlas-label-disclosure";

    const summary = document.createElement("summary");
    summary.className = "atlas-label-summary";

    const textWrap = document.createElement("span");
    textWrap.className = "atlas-label-summary-text";

    const key = document.createElement("span");
    key.className = "atlas-label-key";
    key.textContent = name;
    textWrap.append(key);

    const label = document.createElement("span");
    label.className = "atlas-label-value";
    label.textContent = value;
    textWrap.append(label);

    const count = document.createElement("span");
    count.className = "atlas-label-count";
    count.textContent = `${members.length.toLocaleString()} ${members.length === 1 ? "film" : "films"}`;
    textWrap.append(count);

    const chevron = document.createElement("span");
    chevron.className = "atlas-label-chevron";
    chevron.setAttribute("aria-hidden", "true");

    summary.append(textWrap, chevron);
    details.append(summary);

    const memberList = document.createElement("div");
    memberList.className = "atlas-cluster-members";

    for (const node of members) {
      const button = document.createElement("button");
      button.className = "atlas-cluster-member";
      button.type = "button";
      if (selected?.movie.tmdb_id === node.movie.tmdb_id) {
        button.setAttribute("aria-current", "true");
      }
      button.addEventListener("click", (event) => {
        event.preventDefault();
        selectNode(node, { center: true, zoom: Math.max(zoom, 3.2) });
      });

      const title = document.createElement("span");
      title.className = "atlas-cluster-member-title";
      title.textContent = node.movie.title || node.movie.original_title || "Untitled";
      button.append(title);

      const meta = document.createElement("span");
      meta.className = "atlas-cluster-member-meta";
      meta.textContent = [
        formatYear(node.movie),
        (node.movie.genres ?? []).slice(0, 2).join(", "),
      ].filter(Boolean).join(" / ");
      button.append(meta);

      memberList.append(button);
    }

    details.append(memberList);
    return details;
  };

  const renderSelected = () => {
    if (!selected) {
      selectedTitle.textContent = "The Film Atlas";
      selectedMeta.textContent = `${nodes.length.toLocaleString()} films arranged by semantic distance`;
      selectedOverview.textContent =
        "A static cinema topology built from public movie metadata, layered clustering, and semantic territory layout.";
      selectedLabels.innerHTML = "";
      selectedGenres.innerHTML = "";
      selectedChip.hidden = true;
      selectedChipTitle.textContent = "";
      neighborSection.hidden = true;
      neighborList.innerHTML = "";
      selectedLabels.append(
        renderLabelPill("Macro", `${labelsByLayer.macro.length.toLocaleString()} constellations`),
        renderLabelPill(
          "Neighborhood",
          `${labelsByLayer.neighborhood.length.toLocaleString()} local regions`,
        ),
        renderLabelPill("Micro", `${labelsByLayer.micro.length.toLocaleString()} tight clusters`),
      );
      return;
    }

    const movie = selected.movie;
    const movieTitle = movie.title || movie.original_title || "Untitled";
    selectedTitle.textContent = movieTitle;
    selectedChip.hidden = false;
    selectedChipTitle.textContent = movieTitle;
    selectedChip.setAttribute("aria-label", `Recenter ${movieTitle}`);
    selectedMeta.textContent = [
      formatYear(movie),
      formatRuntime(movie.runtime),
      typeof movie.vote_average === "number" ? `${movie.vote_average.toFixed(1)} TMDb` : "",
    ].filter(Boolean).join(" / ");
    selectedOverview.textContent = movie.overview || "Overview unavailable.";
    neighborSection.hidden = false;

    selectedLabels.innerHTML = "";
    selectedLabels.append(
      renderClusterDisclosure("Macro", selected.macroLabel, "macro", selected.macroId),
      renderClusterDisclosure(
        "Neighborhood",
        selected.neighborhoodLabel,
        "neighborhood",
        selected.neighborhoodId,
      ),
      renderClusterDisclosure("Micro", selected.microLabel, "micro", selected.microId),
    );

    selectedGenres.innerHTML = "";
    for (const genre of (movie.genres ?? []).slice(0, 5)) {
      const pill = document.createElement("span");
      pill.className = "atlas-genre-pill";
      pill.textContent = genre;
      selectedGenres.append(pill);
    }

    neighborList.innerHTML = "";
    const shardId = neighborShardForMovie(movie.tmdb_id);
    if (!loadedNeighborShards.has(shardId)) {
      void loadNeighborShard(movie.tmdb_id).then(() => {
        if (selected?.movie.tmdb_id === movie.tmdb_id) {
          renderSelected();
          requestDraw();
        }
      });
      const loading = document.createElement("p");
      loading.className = "atlas-neighbor-empty";
      loading.textContent = "Loading semantic neighbors...";
      neighborList.append(loading);
      return;
    }

    const record = neighborsById.get(movie.tmdb_id);
    const neighbors = record?.neighbors ?? [];
    for (const neighbor of neighbors.slice(0, 7)) {
      const node = nodesById.get(neighbor.tmdb_id);
      if (!node) continue;

      const button = document.createElement("button");
      button.className = "atlas-neighbor-button";
      button.type = "button";
      button.addEventListener("click", () => selectNode(node, { center: true, zoom: Math.max(zoom, 3.2) }));

      const title = document.createElement("span");
      title.className = "atlas-neighbor-title";
      const neighborTitle = node.movie.title || neighbor.title || "Untitled";
      title.textContent = `${neighborTitle} (${formatYear(node.movie)})`;
      button.append(title);

      const similarity = document.createElement("span");
      similarity.className = "atlas-neighbor-score";
      similarity.textContent = typeof neighbor.similarity === "number"
        ? `${Math.round(neighbor.similarity * 100)}%`
        : "";
      button.append(similarity);

      neighborList.append(button);
    }
  };

  const centerOnNode = (node: AtlasNode, targetZoom = zoom) => {
    const coordinate = coordinateForNode(node);
    zoom = clamp(targetZoom, MIN_ZOOM, MAX_ZOOM);
    offsetX = -(coordinate.x - centerX()) * baseScale * zoom;
    offsetY = (coordinate.y - centerY()) * baseScale * zoom;
  };

  const recenterSelected = () => {
    if (!selected) return;
    centerOnNode(selected, Math.max(zoom, 3.5));
    const coordinate = coordinateForNode(selected);
    const screen = worldToScreen(coordinate.x, coordinate.y);
    moveTooltip(selected, screen);
    requestDraw();
  };

  function selectNode(node: AtlasNode, options: { center?: boolean; zoom?: number } = {}) {
    selected = node;
    hovered = node;
    if (options.center) centerOnNode(node, options.zoom);
    renderSelected();
    const coordinate = coordinateForNode(node);
    const screen = worldToScreen(coordinate.x, coordinate.y);
    moveTooltip(node, screen);
    requestDraw();
  }

  const clearSelection = () => {
    selected = null;
    hovered = null;
    hideTooltip();
    renderSelected();
    requestDraw();
  };

  const zoomAt = (factor: number, x: number, y: number) => {
    const before = screenToWorld(x, y);
    zoom = clamp(zoom * factor, MIN_ZOOM, MAX_ZOOM);
    const after = worldToScreen(before.x, before.y);
    offsetX += x - after.x;
    offsetY += y - after.y;
    requestDraw();
  };

  const resetView = () => {
    selected = null;
    hovered = null;
    zoom = 1;
    offsetX = 0;
    offsetY = 0;
    hideTooltip();
    renderSelected();
    requestDraw();
  };

  const pointerPoint = (event: PointerEvent): ScreenPoint => {
    const rect = canvas.getBoundingClientRect();
    return {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    };
  };

  const updateCanvasCursor = () => {
    if (activePointers.size > 0) {
      canvas.style.cursor = "grabbing";
    } else if (hovered) {
      canvas.style.cursor = "pointer";
    } else {
      canvas.style.cursor = "grab";
    }
  };

  const beginPinchGesture = () => {
    const [first, second] = Array.from(activePointers.values());
    if (!first || !second) return;
    const center = midpoint(first, second);
    gestureStartDistance = Math.max(1, distance(first, second));
    gestureStartZoom = zoom;
    gestureAnchorWorld = screenToWorld(center.x, center.y);
    hasDragged = true;
  };

  const updatePinchGesture = () => {
    const [first, second] = Array.from(activePointers.values());
    if (!first || !second || !gestureAnchorWorld) return;

    const center = midpoint(first, second);
    const scale = distance(first, second) / Math.max(1, gestureStartDistance);
    zoom = clamp(gestureStartZoom * scale, MIN_ZOOM, MAX_ZOOM);
    const after = worldToScreen(gestureAnchorWorld.x, gestureAnchorWorld.y);
    offsetX += center.x - after.x;
    offsetY += center.y - after.y;
    requestDraw();
  };

  const restartDragFromPointer = () => {
    const [remaining] = Array.from(activePointers.values());
    if (!remaining) return;
    dragStartX = remaining.x;
    dragStartY = remaining.y;
    dragOffsetX = offsetX;
    dragOffsetY = offsetY;
    isDragging = true;
  };

  const selectSearchResult = (node: AtlasNode) => {
    searchInput.value = node.movie.title || "";
    searchResults.hidden = true;
    currentSearchResults = [];
    activeSearchIndex = -1;
    selectNode(node, { center: true, zoom: Math.max(zoom, 3.5) });
  };

  const renderSearchResults = (results: AtlasNode[]) => {
    searchResults.innerHTML = "";

    if (!searchInput.value.trim()) {
      searchResults.hidden = true;
      return;
    }

    if (!results.length) {
      const empty = document.createElement("div");
      empty.className = "atlas-search-empty";
      empty.textContent = "No matches";
      searchResults.append(empty);
      searchResults.hidden = false;
      return;
    }

    for (const [index, node] of results.slice(0, 8).entries()) {
      const button = document.createElement("button");
      button.className = "atlas-search-result";
      button.type = "button";
      button.setAttribute("aria-selected", index === activeSearchIndex ? "true" : "false");
      button.addEventListener("click", () => selectSearchResult(node));

      const title = document.createElement("span");
      title.className = "atlas-search-title";
      title.textContent = node.movie.title || "Untitled";
      button.append(title);

      const meta = document.createElement("span");
      meta.className = "atlas-search-meta";
      meta.textContent = `${formatYear(node.movie)} - ${node.mainLabel}`;
      button.append(meta);

      searchResults.append(button);
    }

    searchResults.hidden = false;
  };

  const runSearch = () => {
    const query = normalizeText(searchInput.value.trim());
    if (query.length < 2) {
      currentSearchResults = [];
      activeSearchIndex = -1;
      renderSearchResults([]);
      return [];
    }

    const results = nodes
      .map((node) => ({ node, score: scoreSearch(node, query) }))
      .filter((result) => Number.isFinite(result.score))
      .sort((a, b) => a.score - b.score)
      .map((result) => result.node);

    currentSearchResults = results.slice(0, 8);
    activeSearchIndex = -1;
    renderSearchResults(currentSearchResults);
    return currentSearchResults;
  };

  searchInput.addEventListener("input", runSearch);
  searchInput.addEventListener("focus", runSearch);
  searchInput.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      searchResults.hidden = true;
      return;
    }

    if (event.key !== "ArrowDown" && event.key !== "ArrowUp" && event.key !== "Enter") {
      return;
    }

    if (!currentSearchResults.length) {
      runSearch();
    }
    if (!currentSearchResults.length) return;

    if (event.key === "ArrowDown") {
      event.preventDefault();
      activeSearchIndex = (activeSearchIndex + 1) % currentSearchResults.length;
      renderSearchResults(currentSearchResults);
      return;
    }

    if (event.key === "ArrowUp") {
      event.preventDefault();
      activeSearchIndex = (activeSearchIndex - 1 + currentSearchResults.length) % currentSearchResults.length;
      renderSearchResults(currentSearchResults);
      return;
    }

    if (event.key === "Enter" && activeSearchIndex >= 0) {
      event.preventDefault();
      selectSearchResult(currentSearchResults[activeSearchIndex]);
    }
  });
  searchForm.addEventListener("submit", (event) => {
    event.preventDefault();
    if (!currentSearchResults.length) runSearch();
    const node = currentSearchResults[activeSearchIndex >= 0 ? activeSearchIndex : 0];
    if (node) {
      selectSearchResult(node);
    }
  });

  document.addEventListener("click", (event) => {
    if (!root.contains(event.target as Node)) return;
    if (event.target === searchInput || searchResults.contains(event.target as Node)) return;
    searchResults.hidden = true;
    activeSearchIndex = -1;
  });

  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const modeMultiplier = event.deltaMode === WheelEvent.DOM_DELTA_LINE
      ? 16
      : event.deltaMode === WheelEvent.DOM_DELTA_PAGE
        ? cssHeight
        : 1;
    const delta = event.deltaY * modeMultiplier;
    const factor = Math.exp(-delta * (event.ctrlKey ? 0.006 : 0.0016));
    zoomAt(factor, event.clientX - rect.left, event.clientY - rect.top);
  }, { passive: false });

  canvas.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    canvas.setPointerCapture(event.pointerId);
    const point = pointerPoint(event);
    activePointers.set(event.pointerId, point);
    hideTooltip();

    if (activePointers.size === 2) {
      beginPinchGesture();
      updateCanvasCursor();
      return;
    }

    isDragging = true;
    hasDragged = false;
    dragStartX = point.x;
    dragStartY = point.y;
    dragOffsetX = offsetX;
    dragOffsetY = offsetY;
    updateCanvasCursor();
  });

  canvas.addEventListener("pointermove", (event) => {
    const point = pointerPoint(event);

    if (activePointers.has(event.pointerId)) {
      activePointers.set(event.pointerId, point);
    }

    if (activePointers.size >= 2) {
      updatePinchGesture();
      updateCanvasCursor();
      return;
    }

    if (isDragging && activePointers.has(event.pointerId)) {
      const dx = point.x - dragStartX;
      const dy = point.y - dragStartY;
      if (Math.abs(dx) + Math.abs(dy) > 4) hasDragged = true;
      offsetX = dragOffsetX + dx;
      offsetY = dragOffsetY + dy;
      requestDraw();
      updateCanvasCursor();
      return;
    }

    hovered = findNearest(point.x, point.y);
    if (hovered) {
      moveTooltip(hovered, point);
    } else {
      hideTooltip();
    }
    updateCanvasCursor();
    requestDraw();
  });

  canvas.addEventListener("pointerup", (event) => {
    const point = pointerPoint(event);
    try {
      canvas.releasePointerCapture(event.pointerId);
    } catch {
      // The browser may have already released capture after a touch gesture ends.
    }
    activePointers.delete(event.pointerId);
    isDragging = false;

    if (activePointers.size >= 2) {
      beginPinchGesture();
    } else if (activePointers.size === 1) {
      restartDragFromPointer();
    } else if (!hasDragged) {
      const node = findNearest(point.x, point.y);
      if (node) {
        selectNode(node);
      } else if (selected) {
        clearSelection();
      }
    }

    if (activePointers.size === 0) gestureAnchorWorld = null;
    updateCanvasCursor();
  });

  canvas.addEventListener("pointercancel", (event) => {
    activePointers.delete(event.pointerId);
    isDragging = false;
    gestureAnchorWorld = null;
    updateCanvasCursor();
  });

  canvas.addEventListener("pointerleave", () => {
    if (activePointers.size > 0) return;
    hovered = selected;
    hideTooltip();
    updateCanvasCursor();
    requestDraw();
  });

  resetButton.addEventListener("click", resetView);
  selectedChip.addEventListener("click", recenterSelected);
  zoomInButton.addEventListener("click", () => zoomAt(1.35, cssWidth / 2, cssHeight / 2));
  zoomOutButton.addEventListener("click", () => zoomAt(1 / 1.35, cssWidth / 2, cssHeight / 2));
  layoutSelect?.addEventListener("change", () => setActiveLayout(layoutSelect.value, { resetCamera: true }));
  renderSelect?.addEventListener("change", () => {
    setTerritoryRenderMode(parseTerritoryRenderMode(renderSelect.value));
  });
  colorSelect?.addEventListener("change", () => {
    setColorMode(parseColorMode(colorSelect.value));
  });

  const observer = new ResizeObserver(resizeCanvas);
  observer.observe(stage);

  setStatus("Loading constellation");

  Promise.all([loadAtlasData(dataBase), loadTerritoryLayouts(dataBase)])
    .then(([data, territoryLayouts]) => {
      nodes = buildNodes(data);
      nodesById = new Map(nodes.map((node) => [node.movie.tmdb_id, node]));
      const projectionMode: AtlasLayoutMode = {
        description: "Fallback raw 2D semantic projection used only when territory layouts are unavailable.",
        gmapCells: [],
        id: "projection",
        isTerritory: false,
        label: "Projection fallback",
        pointsById: new Map(),
        regions: [],
      };
      const semanticModes = (territoryLayouts?.variants ?? [])
        .filter((variant) => variant.id === "semantic_gmap_cells")
        .map(buildLayoutMode);
      layoutModes = semanticModes.length > 0 ? semanticModes : [projectionMode];
      const hasGmapLayout = semanticModes.some((mode) => mode.id === "semantic_gmap_cells");
      const initialLayoutId = hasGmapLayout
        ? "semantic_gmap_cells"
        : layoutModes[0]?.id ?? "projection";
      populateColorOptions();
      populateLayoutOptions();
      setActiveLayout(initialLayoutId, { resetCamera: true });
      neighborShardCount = data.manifest.neighbor_shards?.count ?? neighborShardCount;
      neighborShardDirectory = data.manifest.neighbor_shards?.directory ?? neighborShardDirectory;
      movieCount.textContent = (data.manifest.movie_count ?? nodes.length).toLocaleString();
      const totalClusters = (data.manifest.layers?.macro?.cluster_count ?? labelsByLayer.macro.length)
        + (data.manifest.layers?.neighborhood?.cluster_count ?? labelsByLayer.neighborhood.length)
        + (data.manifest.layers?.micro?.cluster_count ?? labelsByLayer.micro.length);
      clusterCount.textContent = totalClusters.toLocaleString();
      generatedAt.textContent = data.manifest.generated_at
        ? new Date(data.manifest.generated_at).toLocaleDateString(undefined, {
          day: "numeric",
          month: "short",
          year: "numeric",
        })
        : "Static export";

      setStatus("", false);
      renderSelected();
      resizeCanvas();
    })
    .catch((error: unknown) => {
      const message = error instanceof Error ? error.message : "Unknown load error";
      setStatus(message);
      console.error(error);
    });
}
