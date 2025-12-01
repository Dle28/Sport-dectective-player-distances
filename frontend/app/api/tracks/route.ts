import { NextResponse } from 'next/server'
import fs from 'fs/promises'
import path from 'path'

export async function GET() {
  try {
    // Prefer freshly generated tracks from the backend run.
    const primaryPath = path.join(
      process.cwd(),
      '..',
      'outputs',
      'run_data',
      'stats',
      'player_tracks_xy.json'
    )
    const fallbackPath = path.join(process.cwd(), '..', 'data', 'tracks_xy', '5ph_demo_tracks.json')

    const loadTracks = async (p: string) => {
      const raw = await fs.readFile(p, 'utf-8')
      const data = JSON.parse(raw)
      const tracks = Object.fromEntries(
        Object.entries<any[]>(data.tracks || {}).map(([pid, points]) => [
          pid,
          points.map((p) => ({ player_id: Number(pid), ...p })),
        ])
      )
      return { ...data, tracks }
    }

    try {
      const data = await loadTracks(primaryPath)
      return NextResponse.json(data)
    } catch {
      // Fallback to sample data if file doesn't exist
      const sampleData = {
        fps: 25.0,
        video_path: 'data/5ph.mp4',
        tracks: {
          '1': [
            { frame_index: 0, player_id: 1, x: 10.0, y: 8.0, visible: true },
            { frame_index: 1, player_id: 1, x: 10.5, y: 8.2, visible: true },
            { frame_index: 2, player_id: 1, x: 11.0, y: 8.5, visible: true },
            { frame_index: 3, player_id: 1, x: 11.5, y: 9.0, visible: true },
          ],
          '2': [
            { frame_index: 0, player_id: 2, x: 20.0, y: 10.0, visible: true },
            { frame_index: 1, player_id: 2, x: 19.5, y: 9.8, visible: true },
            { frame_index: 2, player_id: 2, x: 19.0, y: 9.5, visible: true },
            { frame_index: 3, player_id: 2, x: 18.5, y: 9.0, visible: true },
          ],
        },
      }
      return NextResponse.json(sampleData)
    }
  } catch (error) {
    console.error('Error loading tracks:', error)
    return NextResponse.json({ error: 'Failed to load tracks' }, { status: 500 })
  }
}
