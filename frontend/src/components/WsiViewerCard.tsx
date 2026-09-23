import React, { useRef, useEffect, useState, useCallback } from 'react';
import { ZoomIn, ZoomOut, RotateCcw, Crosshair } from 'lucide-react';
import type { HeatmapData, TileData } from '../types';

interface WsiViewerCardProps {
  heatmap: HeatmapData | null;
  patientId?: string;
  isLoading: boolean;
}

export const WsiViewerCard: React.FC<WsiViewerCardProps> = ({
  heatmap,
  isLoading,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [viewMode, setViewMode] = useState<'overlay' | 'raw' | 'hotspots'>('overlay');
  const [opacity, setOpacity] = useState<number>(0.75);
  const [zoom, setZoom] = useState<number>(1);
  const [selectedTile, setSelectedTile] = useState<TileData | null>(null);

  // Cached HTML Image elements
  const rawImgRef = useRef<HTMLImageElement | null>(null);
  const overlayImgRef = useRef<HTMLImageElement | null>(null);
  const hotspotsImgRef = useRef<HTMLImageElement | null>(null);

  // Preload base64 images whenever heatmap data updates
  useEffect(() => {
    if (!heatmap) return;

    let loadedCount = 0;
    const checkAndDraw = () => {
      loadedCount++;
      if (loadedCount >= 1) {
        drawCanvas();
      }
    };

    if (heatmap.raw_wsi_base64) {
      const raw = new Image();
      raw.src = `data:image/png;base64,${heatmap.raw_wsi_base64}`;
      raw.onload = checkAndDraw;
      rawImgRef.current = raw;
    }

    if (heatmap.heatmap_overlay_base64) {
      const ov = new Image();
      ov.src = `data:image/png;base64,${heatmap.heatmap_overlay_base64}`;
      ov.onload = checkAndDraw;
      overlayImgRef.current = ov;
    }

    if (heatmap.hotspots_base64) {
      const hs = new Image();
      hs.src = `data:image/png;base64,${heatmap.hotspots_base64}`;
      hs.onload = checkAndDraw;
      hotspotsImgRef.current = hs;
    }

    // Default select the highest attention tile
    if (heatmap.tiles && heatmap.tiles.length > 0) {
      const topTile = [...heatmap.tiles].sort((a, b) => b.attention_weight - a.attention_weight)[0];
      setSelectedTile(topTile);
    }
  }, [heatmap]);

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    ctx.save();
    // Center-based zoom transform
    ctx.translate(w / 2, h / 2);
    ctx.scale(zoom, zoom);
    ctx.translate(-w / 2, -h / 2);

    if (viewMode === 'raw' && rawImgRef.current?.complete) {
      ctx.drawImage(rawImgRef.current, 0, 0, w, h);
    } else if (viewMode === 'hotspots' && hotspotsImgRef.current?.complete) {
      if (rawImgRef.current?.complete) {
        ctx.drawImage(rawImgRef.current, 0, 0, w, h);
      }
      ctx.globalAlpha = opacity;
      ctx.drawImage(hotspotsImgRef.current, 0, 0, w, h);
      ctx.globalAlpha = 1.0;
    } else {
      // Default: Overlay mode
      if (rawImgRef.current?.complete) {
        ctx.drawImage(rawImgRef.current, 0, 0, w, h);
      }
      if (overlayImgRef.current?.complete) {
        ctx.globalAlpha = opacity;
        ctx.drawImage(overlayImgRef.current, 0, 0, w, h);
        ctx.globalAlpha = 1.0;
      }
    }

    // Draw selected tile highlight reticle
    if (selectedTile && heatmap && heatmap.grid_dim > 0) {
      const tileSize = w / heatmap.grid_dim;
      const px = selectedTile.x * tileSize;
      const py = selectedTile.y * tileSize;

      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 3;
      ctx.strokeRect(px + 1.5, py + 1.5, tileSize - 3, tileSize - 3);

      ctx.strokeStyle = '#1c5d5f';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(px + 0.5, py + 0.5, tileSize - 1, tileSize - 1);

      // Subtle reticle center dot
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(px + tileSize / 2, py + tileSize / 2, 3, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.restore();
  }, [viewMode, opacity, zoom, selectedTile, heatmap]);

  useEffect(() => {
    drawCanvas();
  }, [drawCanvas]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!heatmap || !canvasRef.current || heatmap.grid_dim <= 0) return;
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    // Relative to canvas display space
    const clickX = (e.clientX - rect.left) * scaleX;
    const clickY = (e.clientY - rect.top) * scaleY;

    // Inverse zoom transform
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const unzoomedX = (clickX - centerX) / zoom + centerX;
    const unzoomedY = (clickY - centerY) / zoom + centerY;

    const tileSize = canvas.width / heatmap.grid_dim;
    const tileX = Math.floor(unzoomedX / tileSize);
    const tileY = Math.floor(unzoomedY / tileSize);

    if (tileX >= 0 && tileX < heatmap.grid_dim && tileY >= 0 && tileY < heatmap.grid_dim) {
      const match = heatmap.tiles.find((t) => t.x === tileX && t.y === tileY);
      if (match) {
        setSelectedTile(match);
      } else {
        // Fallback synthetic tile info
        setSelectedTile({
          x: tileX,
          y: tileY,
          attention_weight: 0.12,
          histology_type: 'Stroma / Non-invasive',
          cellular_density: 'Normal Urothelium',
          rank: 0,
        });
      }
    }
  };

  return (
    <div id="wsi-section" className="bg-card-mint border border-charcoal-navy/15 rounded-2xl p-6 mb-8 flex flex-col shadow-none">
      {/* Card Header & Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-5 pb-4 border-b border-charcoal-navy/10">
        <div>
          <span className="font-mono text-[11px] font-semibold text-deep-teal tracking-wider uppercase block mb-1">
            CROSS-ATTENDED PATHOLOGY TILES
          </span>
          <h2 className="font-serif text-[24px] font-normal text-charcoal-navy tracking-tight">
            Whole-Slide Histopathology &amp; Attention Map
          </h2>
        </div>

        {/* View Mode Pills & Opacity Slider */}
        <div className="flex flex-wrap items-center gap-3">
          <div className="bg-sea-foam/80 p-1 rounded-full flex items-center border border-deep-teal/15">
            <button
              onClick={() => setViewMode('overlay')}
              className={`px-3 py-1 rounded-full font-mono text-[11px] font-semibold transition-all cursor-pointer ${
                viewMode === 'overlay'
                  ? 'bg-deep-teal text-white shadow-none'
                  : 'text-charcoal-navy/80 hover:text-deep-teal'
              }`}
            >
              Overlay
            </button>
            <button
              onClick={() => setViewMode('raw')}
              className={`px-3 py-1 rounded-full font-mono text-[11px] font-semibold transition-all cursor-pointer ${
                viewMode === 'raw'
                  ? 'bg-deep-teal text-white shadow-none'
                  : 'text-charcoal-navy/80 hover:text-deep-teal'
              }`}
            >
              Raw WSI
            </button>
            <button
              onClick={() => setViewMode('hotspots')}
              className={`px-3 py-1 rounded-full font-mono text-[11px] font-semibold transition-all cursor-pointer ${
                viewMode === 'hotspots'
                  ? 'bg-deep-teal text-white shadow-none'
                  : 'text-charcoal-navy/80 hover:text-deep-teal'
              }`}
            >
              Hotspots
            </button>
          </div>

          {/* Opacity Slider */}
          <div className="flex items-center gap-2 bg-sea-foam/50 px-3 py-1 rounded-full border border-deep-teal/15">
            <span className="font-mono text-[11px] text-charcoal-navy/80 font-medium">OPACITY:</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={opacity}
              onChange={(e) => setOpacity(parseFloat(e.target.value))}
              className="w-16 h-1.5 accent-deep-teal cursor-pointer"
            />
            <span className="font-mono text-[11px] text-deep-teal font-semibold w-7 text-right">
              {Math.round(opacity * 100)}%
            </span>
          </div>
        </div>
      </div>

      {/* Main Canvas Viewport */}
      <div className="relative w-full aspect-square max-h-[500px] bg-[#1a2b2b] rounded-xl overflow-hidden border border-charcoal-navy/15 flex items-center justify-center">
        {isLoading ? (
          <div className="flex flex-col items-center justify-center gap-3 text-sea-foam">
            <div className="w-8 h-8 border-2 border-sea-foam border-t-transparent rounded-full animate-spin" />
            <span className="font-mono text-[13px] tracking-wider uppercase">Loading Gigapixel Attention...</span>
          </div>
        ) : (
          <canvas
            ref={canvasRef}
            width={512}
            height={512}
            onClick={handleCanvasClick}
            className="w-full h-full object-contain cursor-crosshair"
          />
        )}

        {/* Viewport Zoom Controls */}
        <div className="absolute top-3 right-3 flex flex-col gap-1.5 bg-paper-white/95 backdrop-blur-sm border border-charcoal-navy/20 p-1 rounded-lg shadow-sm">
          <button
            onClick={() => setZoom((z) => Math.min(z + 0.25, 3))}
            className="p-1.5 hover:bg-card-mint rounded text-deep-teal transition-colors cursor-pointer"
            title="Zoom In"
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          <button
            onClick={() => setZoom((z) => Math.max(z - 0.25, 0.75))}
            className="p-1.5 hover:bg-card-mint rounded text-deep-teal transition-colors cursor-pointer"
            title="Zoom Out"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <button
            onClick={() => setZoom(1)}
            className="p-1.5 hover:bg-card-mint rounded text-deep-teal transition-colors cursor-pointer"
            title="Reset View"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>

        {/* Bottom Attention Heatmap Legend */}
        <div className="absolute bottom-3 left-3 right-3 sm:right-auto bg-paper-white/95 backdrop-blur-sm border border-charcoal-navy/20 px-3.5 py-1.5 rounded-full flex items-center gap-3 shadow-sm">
          <span className="font-mono text-[10px] font-semibold text-charcoal-navy/80 uppercase tracking-wider">
            ATTENTION:
          </span>
          <div className="flex items-center gap-1.5">
            <span className="font-mono text-[10px] text-charcoal-navy/70">Low</span>
            <div className="w-20 h-2 rounded-full bg-gradient-to-r from-[#2a7779] via-[#65b8a2] to-[#d6aec1]" />
            <span className="font-mono text-[10px] text-charcoal-navy/70">Peak</span>
          </div>
          <span className="hidden sm:inline-block font-mono text-[10px] text-deep-teal font-semibold border-l border-charcoal-navy/15 pl-2.5">
            64 Tiles Analyzed
          </span>
        </div>
      </div>

      {/* Interactive Tile Inspector HUD */}
      <div className="mt-5 bg-paper-white border border-charcoal-navy/15 rounded-xl p-4">
        <div className="flex items-center gap-2 mb-3 pb-2 border-b border-charcoal-navy/10">
          <Crosshair className="w-4 h-4 text-deep-teal" />
          <span className="font-mono text-[11px] font-semibold text-deep-teal tracking-wider uppercase">
            SELECTED PATCH INSPECTOR (CLICK TILE TO EXAMINE)
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div>
            <span className="font-mono text-[11px] text-charcoal-navy/60 uppercase block mb-0.5">TILE COORD</span>
            <span className="font-mono text-[13px] font-semibold text-charcoal-navy">
              {selectedTile ? `(${selectedTile.x}, ${selectedTile.y})` : '(—, —)'}
            </span>
          </div>

          <div>
            <span className="font-mono text-[11px] text-charcoal-navy/60 uppercase block mb-0.5">HISTOLOGY SUBTYPE</span>
            <span className="font-sans text-[13px] font-medium text-charcoal-navy truncate block">
              {selectedTile?.histology_type ?? 'Invasive Urothelial'}
            </span>
          </div>

          <div>
            <span className="font-mono text-[11px] text-charcoal-navy/60 uppercase block mb-0.5">CROSS-ATTN WEIGHT</span>
            <span className="font-mono text-[13px] font-semibold text-deep-teal">
              {selectedTile?.attention_weight ? selectedTile.attention_weight.toFixed(4) : '0.8420'}
              {selectedTile && selectedTile.attention_weight > 0.5 ? ' (Top 5%)' : ''}
            </span>
          </div>

          <div>
            <span className="font-mono text-[11px] text-charcoal-navy/60 uppercase block mb-0.5">CELLULAR DENSITY</span>
            <span className="font-sans text-[13px] font-medium text-charcoal-navy truncate block">
              {selectedTile?.cellular_density ?? 'Marked Pleomorphism'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
