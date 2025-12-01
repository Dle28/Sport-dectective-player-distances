import { NextResponse } from 'next/server'
import fs from 'fs/promises'
import path from 'path'

export async function GET() {
  try {
    // Try to read from the parent project's outputs directory
    const filePath = path.join(process.cwd(), '..', 'outputs', 'run_data', 'stats', 'player_distances.json')
    
    try {
      const raw = await fs.readFile(filePath, 'utf-8')
      const data = JSON.parse(raw)
      return NextResponse.json(data)
    } catch {
      // Fallback to sample data if file doesn't exist
      const sampleData = [
        {
          player_id: 1,
          team: 'A',
          total_distance_m: 3250.5,
          total_distance_km: 3.25,
          possession_seconds: 45.2,
        },
        {
          player_id: 2,
          team: 'A',
          total_distance_m: 2980.3,
          total_distance_km: 2.98,
          possession_seconds: 32.1,
        },
        {
          player_id: 3,
          team: 'B',
          total_distance_m: 3100.7,
          total_distance_km: 3.10,
          possession_seconds: 28.5,
        },
      ]
      return NextResponse.json(sampleData)
    }
  } catch (error) {
    console.error('Error loading distances:', error)
    return NextResponse.json({ error: 'Failed to load distances' }, { status: 500 })
  }
}
