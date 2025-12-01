'use client'

import { useEffect, useMemo, useRef } from 'react'
import { useStore } from '@/lib/store'
import { pitchToSvg } from '@/lib/utils'

export default function PitchCanvas() {
  const canvasRef = useRef<SVGSVGElement>(null)
  const { tracks, currentFrame, selectedPlayer, showTrails, heatmapOn } = useStore()

  const pitchMeta = tracks?.pitch_meta
    ? { width: tracks.pitch_meta.length_m, height: tracks.pitch_meta.width_m }
    : { width: 105, height: 68 }

  const SVG_WIDTH = 800
  const SVG_HEIGHT = (pitchMeta.height / pitchMeta.width) * SVG_WIDTH

  // Get all player positions at current frame
  const getCurrentPositions = () => {
    if (!tracks) return []

    const positions: Array<{ playerId: number; x: number; y: number; visible: boolean }> = []

    Object.entries(tracks.tracks).forEach(([id, playerTracks]) => {
      const pid = Number(id)
      const ordered = [...(playerTracks as any[])].sort(
        (a, b) => a.frame_index - b.frame_index
      )

      // Find prev and next samples around currentFrame for interpolation.
      let prev = ordered[0]
      let next = ordered[ordered.length - 1]
      for (let i = 0; i < ordered.length; i++) {
        const t = ordered[i]
        if (t.frame_index <= currentFrame) prev = t
        if (t.frame_index >= currentFrame) {
          next = t
          break
        }
      }

      if (!prev || !next) return
      if (!prev.visible || !next.visible) return

      const span = next.frame_index - prev.frame_index
      const alpha = span > 0 ? (currentFrame - prev.frame_index) / span : 0

      const x = prev.x + (next.x - prev.x) * alpha
      const y = prev.y + (next.y - prev.y) * alpha

      positions.push({
        playerId: pid,
        x,
        y,
        visible: true,
      })
    })

    return positions
  }

  // Get trail positions for selected player
  const getPlayerTrail = () => {
    if (!tracks || selectedPlayer === null || !showTrails) return []
    
    const playerKey = selectedPlayer.toString()
    const playerTracks = tracks.tracks[playerKey]
    
    if (!playerTracks) return []
    
    // Get last 50 frames before current frame
    return (playerTracks as any[])
      .filter(
        (t: any) =>
          t.frame_index <= currentFrame && t.frame_index > currentFrame - 50 && t.visible
      )
      .map((t: any) => pitchToSvg(t.x, t.y, pitchMeta, SVG_WIDTH, SVG_HEIGHT))
  }

  const currentPositions = getCurrentPositions()
  const trail = getPlayerTrail()

  const heatmapPoints = useMemo(() => {
    if (!tracks || !heatmapOn) return []
    const ids = selectedPlayer !== null ? [selectedPlayer.toString()] : Object.keys(tracks.tracks)
    const pts: Array<{ x: number; y: number }> = []
    ids.forEach((pid) => {
      const playerTracks = tracks.tracks[pid] || []
      playerTracks.forEach((t: any) => {
        if (!t.visible) return
        const svgPos = pitchToSvg(t.x, t.y, pitchMeta, SVG_WIDTH, SVG_HEIGHT)
        pts.push(svgPos)
      })
    })
    return pts
  }, [tracks, heatmapOn, selectedPlayer, pitchMeta])

  const hasPositions = currentPositions.length > 0

  return (
    <div className="w-full h-full min-h-[480px] flex items-center justify-center bg-pitch-grass p-4">
      <svg
        ref={canvasRef}
        viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`}
        width="100%"
        height="100%"
        preserveAspectRatio="xMidYMid meet"
        className="w-full h-full max-h-full border-2 border-white"
      >
        {/* Pitch background */}
        <rect x="0" y="0" width={SVG_WIDTH} height={SVG_HEIGHT} fill="#1a5f3e" />
        
        {/* Center circle */}
        <circle
          cx={SVG_WIDTH / 2}
          cy={SVG_HEIGHT / 2}
          r={(9.15 / pitchMeta.width) * SVG_WIDTH}
          fill="none"
          stroke="white"
          strokeWidth="2"
        />
        
        {/* Center line */}
        <line
          x1={SVG_WIDTH / 2}
          y1="0"
          x2={SVG_WIDTH / 2}
          y2={SVG_HEIGHT}
          stroke="white"
          strokeWidth="2"
        />
        
        {/* Penalty areas */}
        <rect
          x="0"
          y={(SVG_HEIGHT - (40.3 / pitchMeta.height) * SVG_HEIGHT) / 2}
          width={(16.5 / pitchMeta.width) * SVG_WIDTH}
          height={(40.3 / pitchMeta.height) * SVG_HEIGHT}
          fill="none"
          stroke="white"
          strokeWidth="2"
        />
        <rect
          x={SVG_WIDTH - (16.5 / pitchMeta.width) * SVG_WIDTH}
          y={(SVG_HEIGHT - (40.3 / pitchMeta.height) * SVG_HEIGHT) / 2}
          width={(16.5 / pitchMeta.width) * SVG_WIDTH}
          height={(40.3 / pitchMeta.height) * SVG_HEIGHT}
          fill="none"
          stroke="white"
          strokeWidth="2"
        />
        
        {/* Goal areas */}
        <rect
          x="0"
          y={(SVG_HEIGHT - (18.3 / pitchMeta.height) * SVG_HEIGHT) / 2}
          width={(5.5 / pitchMeta.width) * SVG_WIDTH}
          height={(18.3 / pitchMeta.height) * SVG_HEIGHT}
          fill="none"
          stroke="white"
          strokeWidth="2"
        />
        <rect
          x={SVG_WIDTH - (5.5 / pitchMeta.width) * SVG_WIDTH}
          y={(SVG_HEIGHT - (18.3 / pitchMeta.height) * SVG_HEIGHT) / 2}
          width={(5.5 / pitchMeta.width) * SVG_WIDTH}
          height={(18.3 / pitchMeta.height) * SVG_HEIGHT}
          fill="none"
          stroke="white"
          strokeWidth="2"
        />

        {/* Heatmap (dense points, low opacity) */}
        {heatmapOn &&
          heatmapPoints.map((p, idx) => (
            <circle
              key={`heat-${idx}`}
              cx={p.x}
              cy={p.y}
              r={6}
              fill="#f97316"
              opacity={0.08}
            />
          ))}

        {/* Trail for selected player */}
        {trail.length > 1 && (
          <polyline
            points={trail.map((p) => `${p.x},${p.y}`).join(' ')}
            fill="none"
            stroke="#fbbf24"
            strokeWidth="2"
            opacity="0.6"
          />
        )}
        {/* If no positions, show a subtle center mark so the pitch isn't empty */}
        {!hasPositions && (
          <text
            x={SVG_WIDTH / 2}
            y={SVG_HEIGHT / 2}
            fill="#ffffff"
            fontSize="18"
            textAnchor="middle"
            opacity="0.5"
          >
            Waiting for track data...
          </text>
        )}

        {/* Player positions */}
        {currentPositions.map((pos) => {
          const svgPos = pitchToSvg(pos.x, pos.y, pitchMeta, SVG_WIDTH, SVG_HEIGHT)
          const isSelected = pos.playerId === selectedPlayer
          const color = isSelected ? '#fbbf24' : pos.playerId % 2 === 0 ? '#3b82f6' : '#ef4444'
          
          return (
            <g key={pos.playerId}>
              <circle
                cx={svgPos.x}
                cy={svgPos.y}
                r={isSelected ? 8 : 6}
                fill={color}
                stroke="white"
                strokeWidth={isSelected ? 3 : 2}
                style={{ cursor: 'pointer' }}
                onClick={() => useStore.getState().setSelectedPlayer(pos.playerId)}
              />
              <text
                x={svgPos.x}
                y={svgPos.y - 12}
                fill="white"
                fontSize="12"
                fontWeight="bold"
                textAnchor="middle"
              >
                {pos.playerId}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}
