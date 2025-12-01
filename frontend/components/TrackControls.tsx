'use client'

import { useStore } from '@/lib/store'
import { frameToTime } from '@/lib/utils'
import { useEffect } from 'react'

export default function TrackControls() {
  const { tracks, currentFrame, playing, selectedPlayer, setCurrentFrame, setPlaying, setSelectedPlayer, reset } = useStore()

  const fps = tracks?.fps ?? 25
  const maxFrame = tracks
    ? Math.max(...Object.values(tracks.tracks).flatMap((t) => t.map((p) => p.frame_index)))
    : 0

  const playerIds = tracks ? Object.keys(tracks.tracks).map(Number) : []

  // Auto-advance frame when playing
  useEffect(() => {
    if (!playing) return

    const interval = setInterval(() => {
      setCurrentFrame(currentFrame + 1 >= maxFrame ? 0 : currentFrame + 1)
    }, 1000 / fps)

    return () => clearInterval(interval)
  }, [playing, currentFrame, maxFrame, fps, setCurrentFrame])

  return (
    <div className="space-y-4">
      <div>
        <label className="block text-sm font-medium mb-2">Select Player</label>
        <select
          value={selectedPlayer ?? ''}
          onChange={(e) => setSelectedPlayer(e.target.value ? Number(e.target.value) : null)}
          className="w-full px-3 py-2 border rounded-md bg-white"
        >
          <option value="">All Players</option>
          {playerIds.map((id) => (
            <option key={id} value={id}>
              Player {id}
            </option>
          ))}
        </select>
      </div>

      <div>
        <div className="flex items-center justify-between mb-2">
          <label className="text-sm font-medium">Timeline</label>
          <span className="text-sm text-gray-600">
            {frameToTime(currentFrame, fps)} / {frameToTime(maxFrame, fps)}
          </span>
        </div>
        <input
          type="range"
          min="0"
          max={maxFrame}
          value={currentFrame}
          onChange={(e) => setCurrentFrame(Number(e.target.value))}
          className="w-full"
        />
        <div className="text-xs text-gray-500 mt-1">Frame: {currentFrame}</div>
      </div>

      <div className="flex gap-2">
        <button
          onClick={() => setPlaying(!playing)}
          className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 font-medium"
        >
          {playing ? 'Pause' : 'Play'}
        </button>
        <button
          onClick={() => reset()}
          className="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700"
        >
          Reset
        </button>
      </div>
    </div>
  )
}
