'use client'

import { useEffect } from 'react'
import PitchCanvas from '@/components/PitchCanvas'
import TrackControls from '@/components/TrackControls'
import HeatmapToggle from '@/components/HeatmapToggle'
import DistanceTable from '@/components/DistanceTable'
import { useStore } from '@/lib/store'

export default function Home() {
  const { tracks, distances, setTracks, setDistances } = useStore()

  useEffect(() => {
    // Load tracks data
    fetch('/api/tracks')
      .then((res) => res.json())
      .then((data) => setTracks(data))
      .catch((err) => console.error('Failed to load tracks:', err))

    // Load distances data
    fetch('/api/distances')
      .then((res) => res.json())
      .then((data) => setDistances(data))
      .catch((err) => console.error('Failed to load distances:', err))
  }, [setTracks, setDistances])

  if (!tracks || !distances) {
    return (
      <main className="flex min-h-screen flex-col items-center justify-center p-24 bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
        <div className="text-center fade-in">
          <div className="loading-spinner mx-auto mb-6"></div>
          <div className="text-3xl font-bold text-white mb-2">⚽ Loading Match Data</div>
          <div className="text-gray-400">Analyzing player movements...</div>
        </div>
      </main>
    )
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 p-4 md:p-8">
      {/* Header */}
      <div className="mb-8 fade-in">
        <div className="flex items-center gap-4 mb-2">
          <div className="text-5xl">⚽</div>
          <div>
            <h1 className="text-4xl md:text-5xl font-bold text-gradient">
              Football Analytics
            </h1>
            <p className="text-gray-400 text-sm md:text-base mt-1">Real-time Player Tracking & Performance Dashboard</p>
          </div>
        </div>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 md:gap-6">
        {/* Main pitch visualization */}
        <div className="lg:col-span-2 glass-card rounded-2xl p-4 md:p-6 shadow-2xl slide-in">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
              <span className="text-3xl">🏟️</span>
              Live Match View
            </h2>
            <div className="flex items-center gap-2 text-sm">
              <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-gray-600 font-medium">LIVE</span>
            </div>
          </div>
          <PitchCanvas />
        </div>

        {/* Controls and stats sidebar */}
        <div className="space-y-4 md:space-y-6">
          <div className="glass-card rounded-2xl p-4 md:p-6 shadow-2xl slide-in">
            <h2 className="text-xl md:text-2xl font-bold mb-4 text-gray-800 flex items-center gap-2">
              <span className="text-2xl">🎮</span>
              Match Controls
            </h2>
            <TrackControls />
            <div className="mt-6 pt-6 border-t border-gray-200">
              <HeatmapToggle />
            </div>
          </div>

          <div className="glass-card rounded-2xl p-4 md:p-6 shadow-2xl slide-in">
            <h2 className="text-xl md:text-2xl font-bold mb-4 text-gray-800 flex items-center gap-2">
              <span className="text-2xl">📊</span>
              Player Statistics
            </h2>
            <DistanceTable />
          </div>
        </div>
      </div>
    </main>
  )
}
