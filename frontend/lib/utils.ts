import { PitchMeta } from './types'

/**
 * Scale pitch coordinates (meters) to SVG pixel coordinates
 */
export function pitchToSvg(
  x: number,
  y: number,
  pitchMeta: PitchMeta,
  svgWidth: number,
  svgHeight: number
): { x: number; y: number } {
  const sx = svgWidth / pitchMeta.width
  const sy = svgHeight / pitchMeta.height
  return { x: x * sx, y: y * sy }
}

/**
 * Format time from frame index and FPS
 */
export function frameToTime(frameIndex: number, fps: number): string {
  const totalSeconds = frameIndex / fps
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = Math.floor(totalSeconds % 60)
  return `${minutes}:${seconds.toString().padStart(2, '0')}`
}

/**
 * Get color for player ID
 */
export function getPlayerColor(playerId: number): string {
  const colors = [
    '#3b82f6', // blue
    '#ef4444', // red
    '#10b981', // green
    '#f59e0b', // amber
    '#8b5cf6', // purple
    '#ec4899', // pink
    '#14b8a6', // teal
    '#f97316', // orange
  ]
  return colors[playerId % colors.length]
}
