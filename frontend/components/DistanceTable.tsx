'use client'

import { useStore } from '@/lib/store'
import { useMemo, useState } from 'react'

export default function DistanceTable() {
  const { distances, selectedPlayer, setSelectedPlayer } = useStore()
  const [limit, setLimit] = useState(50)
  const [showAll, setShowAll] = useState(false)

  const sortedDistances = useMemo(() => {
    if (!distances) return []
    return [...distances].sort((a, b) => b.total_distance_m - a.total_distance_m)
  }, [distances])

  const visibleRows = useMemo(() => {
    if (showAll) return sortedDistances
    return sortedDistances.slice(0, limit)
  }, [sortedDistances, showAll, limit])

  const totalCount = sortedDistances.length

  return (
    <div className="space-y-3">
      <h3 className="text-lg font-semibold">Player Statistics</h3>
      <div className="text-xs text-gray-500">
        Showing {visibleRows.length} of {totalCount} players
      </div>
      
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b-2 border-gray-300">
              <th className="text-left py-2 px-2">Player</th>
              <th className="text-left py-2 px-2">Team</th>
              <th className="text-right py-2 px-2">Distance (m)</th>
              <th className="text-right py-2 px-2">Distance (km)</th>
              <th className="text-right py-2 px-2">Possession (s)</th>
            </tr>
          </thead>
          <tbody>
            {visibleRows.map((player) => {
              const isSelected = player.player_id === selectedPlayer
              return (
                <tr
                  key={player.player_id}
                  onClick={() => setSelectedPlayer(player.player_id)}
                  className={`border-b border-gray-200 cursor-pointer hover:bg-gray-100 ${
                    isSelected ? 'bg-yellow-100 font-semibold' : ''
                  }`}
                >
                  <td className="py-2 px-2">{player.player_id}</td>
                  <td className="py-2 px-2">
                    <span
                      className={`inline-block w-3 h-3 rounded-full mr-1 ${
                        player.team === 'Team A' ? 'bg-blue-500' : 'bg-red-500'
                      }`}
                    />
                    {player.team}
                  </td>
                  <td className="text-right py-2 px-2">{player.total_distance_m.toFixed(2)}</td>
                  <td className="text-right py-2 px-2">{player.total_distance_km.toFixed(4)}</td>
                  <td className="text-right py-2 px-2">{player.possession_seconds.toFixed(1)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <div className="flex items-center justify-between">
        {!showAll && totalCount > limit && (
          <button
            onClick={() => setShowAll(true)}
            className="px-3 py-2 text-xs font-semibold text-white bg-blue-600 rounded hover:bg-blue-700"
          >
            Show all ({totalCount})
          </button>
        )}
        {showAll && (
          <button
            onClick={() => setShowAll(false)}
            className="px-3 py-2 text-xs font-semibold text-white bg-gray-600 rounded hover:bg-gray-700"
          >
            Show top {limit}
          </button>
        )}
        {!showAll && totalCount > limit && (
          <div className="text-xs text-gray-500">Limited for readability</div>
        )}
      </div>

      {sortedDistances.length === 0 && (
        <div className="text-center text-gray-500 py-4">No distance data available</div>
      )}
    </div>
  )
}
