import { create } from 'zustand'
import { TracksData, PlayerDistance, PitchMeta } from './types'

type TrackingState = {
  tracks: TracksData | null
  distances: PlayerDistance[]
  pitchMeta: PitchMeta
  selectedPlayer: number | null
  currentFrame: number
  playing: boolean
  heatmapOn: boolean
  showTrails: boolean
  isLoading: boolean
  
  // Actions
  setTracks: (tracks: TracksData) => void
  setDistances: (distances: PlayerDistance[]) => void
  setSelectedPlayer: (playerId: number | null) => void
  setCurrentFrame: (frame: number) => void
  setPlaying: (playing: boolean) => void
  setHeatmapOn: (on: boolean) => void
  setShowTrails: (show: boolean) => void
  reset: () => void
}

export const useStore = create<TrackingState>((set) => ({
  tracks: null,
  distances: [],
  pitchMeta: {
    width: 105.0,
    height: 68.0,
  },
  selectedPlayer: null,
  currentFrame: 0,
  playing: false,
  heatmapOn: false,
  showTrails: true,
  isLoading: false,

  setTracks: (tracks) => set({ tracks }),
  setDistances: (distances) => set({ distances }),
  setSelectedPlayer: (playerId) => set({ selectedPlayer: playerId }),
  setCurrentFrame: (frame) => set({ currentFrame: frame }),
  setPlaying: (playing) => set({ playing }),
  setHeatmapOn: (on) => set({ heatmapOn: on }),
  setShowTrails: (show) => set({ showTrails: show }),
  reset: () =>
    set({
      selectedPlayer: null,
      currentFrame: 0,
      playing: false,
    }),
}))
