import type {
  FilmAtlasCluster,
  FilmAtlasExport,
  FilmAtlasLabel,
  FilmAtlasLayer,
  FilmAtlasMovie,
  FilmAtlasNeighborRecord,
  FilmAtlasPoint,
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

type LabelBox = {
  bottom: number;
  left: number;
  right: number;
  top: number;
};

const FILES: Record<keyof FilmAtlasExport, string> = {
  labels: "labels.json",
  macro_clusters: "macro_clusters.json",
  manifest: "manifest.json",
  micro_clusters: "micro_clusters.json",
  movies: "movies.json",
  neighborhood_clusters: "neighborhood_clusters.json",
  neighbors: "neighbors.json",
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

const MIN_ZOOM = 0.72;
const MAX_ZOOM = 10;
const LABEL_TRANSITION_MS = 280;

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

const distance = (a: ScreenPoint, b: ScreenPoint) => Math.hypot(a.x - b.x, a.y - b.y);

const midpoint = (a: ScreenPoint, b: ScreenPoint): ScreenPoint => ({
  x: (a.x + b.x) / 2,
  y: (a.y + b.y) / 2,
});

const boxesOverlap = (a: LabelBox, b: LabelBox, gap: number) =>
  a.left - gap < b.right
  && a.right + gap > b.left
  && a.top - gap < b.bottom
  && a.bottom + gap > b.top;

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

async function fetchJson<T>(baseUrl: string, fileName: string): Promise<T> {
  const response = await fetch(`${baseUrl.replace(/\/$/, "")}/${fileName}`, {
    cache: "force-cache",
  });

  if (!response.ok) {
    throw new Error(`Could not load ${fileName}: ${response.status}`);
  }

  return response.json() as Promise<T>;
}

async function loadAtlasData(baseUrl: string): Promise<FilmAtlasExport> {
  const entries = await Promise.all(
    (Object.entries(FILES) as Array<[keyof FilmAtlasExport, string]>).map(
      async ([key, file]) => [key, await fetchJson<FilmAtlasExport[typeof key]>(baseUrl, file)] as const,
    ),
  );

  return Object.fromEntries(entries) as FilmAtlasExport;
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

function buildNodes(data: FilmAtlasExport) {
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

function getBounds(nodes: AtlasNode[]): Bounds {
  return nodes.reduce(
    (bounds, node) => ({
      maxX: Math.max(bounds.maxX, node.point.x),
      maxY: Math.max(bounds.maxY, node.point.y),
      minX: Math.min(bounds.minX, node.point.x),
      minY: Math.min(bounds.minY, node.point.y),
    }),
    { maxX: -Infinity, maxY: -Infinity, minX: Infinity, minY: Infinity },
  );
}

function buildClusterLabels(nodes: AtlasNode[], layer: FilmAtlasLayer): ClusterLabel[] {
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
    current.x += node.point.x;
    current.y += node.point.y;
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
  if (scale < 1.85) return "macro";
  if (scale < 4.2) return "neighborhood";
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
  const selectedTitle = getElement<HTMLElement>(root, "[data-atlas-selected-title]");
  const selectedMeta = getElement<HTMLElement>(root, "[data-atlas-selected-meta]");
  const selectedOverview = getElement<HTMLElement>(root, "[data-atlas-selected-overview]");
  const selectedLabels = getElement<HTMLElement>(root, "[data-atlas-selected-labels]");
  const selectedGenres = getElement<HTMLElement>(root, "[data-atlas-selected-genres]");
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
  let neighborsById = new Map<number, FilmAtlasNeighborRecord>();
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
    }
  };

  const labelTransitionProgress = () => {
    if (!previousLabelLayer) return 1;
    return clamp((performance.now() - labelTransitionStartedAt) / LABEL_TRANSITION_MS, 0, 1);
  };

  const fitToView = () => {
    const width = Math.max(1, bounds.maxX - bounds.minX);
    const height = Math.max(1, bounds.maxY - bounds.minY);
    const padding = Math.max(34, Math.min(cssWidth, cssHeight) * 0.1);
    baseScale = Math.min((cssWidth - padding * 2) / width, (cssHeight - padding * 2) / height);
    if (!Number.isFinite(baseScale) || baseScale <= 0) baseScale = 1;

    if (!cameraInitialized) {
      zoom = 1;
      offsetX = 0;
      offsetY = 0;
      cameraInitialized = true;
    }
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

  const drawNeighborLines = () => {
    if (!selected) return;
    const record = neighborsById.get(selected.movie.tmdb_id);
    if (!record?.neighbors?.length) return;

    const from = worldToScreen(selected.point.x, selected.point.y);
    ctx.save();
    ctx.lineWidth = 1;
    for (const neighbor of record.neighbors.slice(0, 8)) {
      const node = nodesById.get(neighbor.tmdb_id);
      if (!node) continue;
      const to = worldToScreen(node.point.x, node.point.y);
      ctx.globalAlpha = 0.18 + clamp((neighbor.similarity ?? 0.5) - 0.45, 0, 0.25);
      ctx.strokeStyle = selected.color;
      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.stroke();
    }
    ctx.restore();
  };

  const drawPoints = () => {
    const baseRadius = clamp(1.05 + Math.log2(zoom + 1) * 0.34, 1.05, 2.6);

    ctx.save();
    for (const node of nodes) {
      const screen = worldToScreen(node.point.x, node.point.y);
      if (screen.x < -20 || screen.x > cssWidth + 20 || screen.y < -20 || screen.y > cssHeight + 20) {
        continue;
      }

      const isSelected = selected?.movie.tmdb_id === node.movie.tmdb_id;
      const isHovered = hovered?.movie.tmdb_id === node.movie.tmdb_id;
      const radius = isSelected ? baseRadius + 3.9 : isHovered ? baseRadius + 2.4 : baseRadius;

      ctx.globalAlpha = isSelected || isHovered ? 1 : 0.72;
      ctx.fillStyle = node.color;
      ctx.beginPath();
      ctx.arc(screen.x, screen.y, radius, 0, Math.PI * 2);
      ctx.fill();

      if (isSelected || isHovered) {
        ctx.globalAlpha = isSelected ? 0.46 : 0.26;
        ctx.lineWidth = isSelected ? 2 : 1.5;
        ctx.strokeStyle = node.color;
        ctx.beginPath();
        ctx.arc(screen.x, screen.y, radius + 5, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
    ctx.restore();
  };

  const drawLabelSet = (layer: FilmAtlasLayer, alpha: number) => {
    const labels = labelsByLayer[layer];
    const maxLabels = layer === "macro" ? 12 : layer === "neighborhood" ? (zoom < 2.6 ? 20 : 30) : 34;
    const fontSize = layer === "macro" ? 14 : layer === "neighborhood" ? 11.5 : 9.5;
    const lineHeight = fontSize * (layer === "macro" ? 1.08 : 1.16);
    const collisionGap = layer === "macro" ? 8 : layer === "neighborhood" ? 5 : 3;

    ctx.save();
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.font = `${fontSize}px Inter, ui-sans-serif, system-ui`;

    let drawn = 0;
    const placed: LabelBox[] = [];
    for (const label of labels) {
      if (drawn >= maxLabels) break;
      const screen = worldToScreen(label.x, label.y);
      const lines = getLabelLines(label.label, layer);
      const width = Math.max(...lines.map((line) => ctx.measureText(line).width));
      const height = lines.length * lineHeight;
      const box = {
        bottom: screen.y + height / 2 + 4,
        left: screen.x - width / 2 - 8,
        right: screen.x + width / 2 + 8,
        top: screen.y - height / 2 - 4,
      };

      if (box.left < 8 || box.right > cssWidth - 8 || box.top < 8 || box.bottom > cssHeight - 8) {
        continue;
      }

      if (placed.some((placedBox) => boxesOverlap(box, placedBox, collisionGap))) {
        continue;
      }

      ctx.lineWidth = layer === "macro" ? 7 : 5;
      ctx.strokeStyle = "rgba(5, 5, 5, 0.82)";
      ctx.globalAlpha = alpha * (layer === "macro" ? 0.86 : 0.72);
      ctx.fillStyle = layer === "macro" ? label.color : "#f7ead2";
      lines.forEach((line, lineIndex) => {
        const y = screen.y + (lineIndex - (lines.length - 1) / 2) * lineHeight;
        ctx.strokeText(line, screen.x, y);
        ctx.fillText(line, screen.x, y);
      });
      ctx.globalAlpha = 1;
      placed.push(box);
      drawn += 1;
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
    drawNeighborLines();
    drawPoints();
    drawLabels();
  }

  const findNearest = (x: number, y: number) => {
    let nearest: AtlasNode | null = null;
    let nearestDistance = Infinity;
    const hitRadius = clamp(10 + zoom * 1.5, 10, 18);

    for (const node of nodes) {
      const screen = worldToScreen(node.point.x, node.point.y);
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
    chevron.textContent = "v";

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
        "A static cinema topology built from public movie metadata, layered clustering, and UMAP projection.";
      selectedLabels.innerHTML = "";
      selectedGenres.innerHTML = "";
      neighborList.innerHTML = "";
      selectedLabels.append(
        renderLabelPill("Macro", "12 constellations"),
        renderLabelPill("Neighborhood", "75 local regions"),
        renderLabelPill("Micro", "200 tight clusters"),
      );
      return;
    }

    const movie = selected.movie;
    selectedTitle.textContent = movie.title || movie.original_title || "Untitled";
    selectedMeta.textContent = [
      formatYear(movie),
      formatRuntime(movie.runtime),
      typeof movie.vote_average === "number" ? `${movie.vote_average.toFixed(1)} TMDb` : "",
    ].filter(Boolean).join(" / ");
    selectedOverview.textContent = movie.overview || "Overview unavailable.";

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
      title.textContent = node.movie.title || neighbor.title || "Untitled";
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
    zoom = clamp(targetZoom, MIN_ZOOM, MAX_ZOOM);
    offsetX = -(node.point.x - centerX()) * baseScale * zoom;
    offsetY = (node.point.y - centerY()) * baseScale * zoom;
  };

  function selectNode(node: AtlasNode, options: { center?: boolean; zoom?: number } = {}) {
    selected = node;
    hovered = node;
    if (options.center) centerOnNode(node, options.zoom);
    renderSelected();
    const screen = worldToScreen(node.point.x, node.point.y);
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

    for (const node of results.slice(0, 8)) {
      const button = document.createElement("button");
      button.className = "atlas-search-result";
      button.type = "button";
      button.addEventListener("click", () => {
        searchInput.value = node.movie.title || "";
        searchResults.hidden = true;
        selectNode(node, { center: true, zoom: Math.max(zoom, 3.5) });
      });

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
      renderSearchResults([]);
      return [];
    }

    const results = nodes
      .map((node) => ({ node, score: scoreSearch(node, query) }))
      .filter((result) => Number.isFinite(result.score))
      .sort((a, b) => a.score - b.score)
      .map((result) => result.node);

    renderSearchResults(results);
    return results;
  };

  searchInput.addEventListener("input", runSearch);
  searchInput.addEventListener("focus", runSearch);
  searchForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const [first] = runSearch();
    if (first) {
      searchResults.hidden = true;
      selectNode(first, { center: true, zoom: Math.max(zoom, 3.5) });
    }
  });

  document.addEventListener("click", (event) => {
    if (!root.contains(event.target as Node)) return;
    if (event.target === searchInput || searchResults.contains(event.target as Node)) return;
    searchResults.hidden = true;
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
  zoomInButton.addEventListener("click", () => zoomAt(1.35, cssWidth / 2, cssHeight / 2));
  zoomOutButton.addEventListener("click", () => zoomAt(1 / 1.35, cssWidth / 2, cssHeight / 2));

  const observer = new ResizeObserver(resizeCanvas);
  observer.observe(stage);

  setStatus("Loading constellation");

  loadAtlasData(dataBase)
    .then((data) => {
      nodes = buildNodes(data);
      nodesById = new Map(nodes.map((node) => [node.movie.tmdb_id, node]));
      neighborsById = new Map(data.neighbors.map((record) => [record.tmdb_id, record]));
      bounds = getBounds(nodes);
      labelsByLayer = {
        macro: buildClusterLabels(nodes, "macro"),
        micro: buildClusterLabels(nodes, "micro"),
        neighborhood: buildClusterLabels(nodes, "neighborhood"),
      };
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
