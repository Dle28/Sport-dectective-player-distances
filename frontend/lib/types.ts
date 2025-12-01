export type PlayerPitchPoint = {
  frame_index: number;
  player_id: number;
  x: number;
  y: number;
  visible: boolean;
};

export type PitchMeta = {
  width: number;
  height: number;
};

export type TracksData = {
  fps: number;
  video_path: string | null;
  pitch_meta?: { pitch_type: string; length_m: number; width_m: number; description?: string | null };
  tracks: Record<string, PlayerPitchPoint[]>;
};

export type PlayerDistance = {
  player_id: number;
  team: string;
  total_distance_m: number;
  total_distance_km: number;
  possession_seconds: number;
};
