'use client'

import { useStore } from '@/lib/store'

export default function HeatmapToggle() {
  const { heatmapOn, showTrails, setHeatmapOn, setShowTrails } = useStore()

  return (
    <div className="space-y-3">
      <h3 className="text-lg font-semibold">Visualization Options</h3>
      
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium">Show Heatmap</label>
        <button
          onClick={() => setHeatmapOn(!heatmapOn)}
          className={`px-4 py-2 rounded-md font-medium transition-colors ${
            heatmapOn
              ? 'bg-green-600 text-white hover:bg-green-700'
              : 'bg-gray-300 text-gray-700 hover:bg-gray-400'
          }`}
        >
          {heatmapOn ? 'ON' : 'OFF'}
        </button>
      </div>

      <div className="flex items-center justify-between">
        <label className="text-sm font-medium">Show Trails</label>
        <button
          onClick={() => setShowTrails(!showTrails)}
          className={`px-4 py-2 rounded-md font-medium transition-colors ${
            showTrails
              ? 'bg-yellow-600 text-white hover:bg-yellow-700'
              : 'bg-gray-300 text-gray-700 hover:bg-gray-400'
          }`}
        >
          {showTrails ? 'ON' : 'OFF'}
        </button>
      </div>
    </div>
  )
}
