// Track definitions. Angles: + = left turn, - = right turn. Heading 90 deg = +X (east), 0 = +Z (south).
import type { TrackDef } from './track';

/**
 * 赤霞湾国际赛道 — a clockwise coastal circuit: long pit straight, a heavy-braking first corner,
 * a fast downhill sweeper into an infield hairpin, a flat-out run along the sea wall,
 * then a climb to a summit hairpin and downhill esses back to the line.
 */
export const CRIMSON_BAY: TrackDef = {
  id: 'crimson_bay',
  name: 'CRIMSON BAY CIRCUIT',
  nameCn: '赤霞湾国际赛道',
  blurb: '海岸 · 山道 · 3.5 KM · 14 弯',
  heading0: 90,
  y0: 16,
  halfWidth: 7.0,
  bankGain: 3.2,
  bankMax: 0.115,
  seed: 20261,
  segs: [
    { kind: 'S', len: 430, y: 16, hw: 8.0, adjust: 'a', tag: 'sf_out' }, // 0  start line -> T1 braking zone
    { kind: 'A', r: 46, deg: -90, y: 15, tag: 'T1' }, // 1
    { kind: 'S', len: 70, y: 14.5 }, // 2
    { kind: 'A', r: 62, deg: 35, y: 14, tag: 'T2' }, // 3
    { kind: 'S', len: 60, y: 13.2 }, // 4
    { kind: 'A', r: 160, deg: -125, y: 9, tag: 'T3' }, // 5  downhill sweeper
    { kind: 'S', len: 150, y: 8.4 }, // 6
    { kind: 'A', r: 30, deg: 180, y: 8, hw: 7.6, tag: 'T4' }, // 7  infield hairpin
    { kind: 'S', len: 130, y: 7.4 }, // 8
    { kind: 'A', r: 56, deg: -90, y: 7, tag: 'T5' }, // 9
    { kind: 'S', len: 40, y: 6.4 }, // 10
    { kind: 'A', r: 100, deg: -90, y: 5.4, tag: 'T6' }, // 11 onto the coast
    { kind: 'S', len: 210, y: 5, hw: 7.4, tag: 'coast' }, // 12
    { kind: 'A', r: 700, deg: 7, y: 5 }, // 13 flat-out kink
    { kind: 'A', r: 520, deg: -14, y: 5, tag: 'T7' }, // 14
    { kind: 'A', r: 700, deg: 7, y: 5.2 }, // 15
    { kind: 'S', len: 170, y: 5.6, hw: 7.4 }, // 16
    { kind: 'A', r: 70, deg: -90, y: 7, tag: 'T8' }, // 17
    { kind: 'S', len: 200, y: 23, adjust: 'b', tag: 'climb' }, // 18
    { kind: 'A', r: 110, deg: 40, y: 27, tag: 'T9' }, // 19
    { kind: 'S', len: 60, y: 29.2 }, // 20
    { kind: 'A', r: 34, deg: -150, y: 30, hw: 7.6, tag: 'T10' }, // 21 summit hairpin
    { kind: 'S', len: 40, y: 29 }, // 22
    { kind: 'A', r: 72, deg: 110, y: 24.5, tag: 'T11' }, // 23 downhill esses
    { kind: 'S', len: 60, y: 22 }, // 24
    { kind: 'A', r: 104, deg: -90, y: 17, tag: 'T12' }, // 25 final corner
    { kind: 'S', len: 250, y: 16, hw: 8.0, tag: 'sf_in' }, // 26 back to the line
  ],
  zones: [
    { kind: 'grandstand', s0: -235, s1: 150, side: 'L', offset: 9 },
    { kind: 'pit', s0: -215, s1: 215, side: 'R', offset: 7 },
    { kind: 'lights', s0: -260, s1: 440, side: 'both', offset: 5 },
    { kind: 'grandstand', s0: 470, s1: 560, side: 'L', offset: 22 },
    { kind: 'billboards', s0: 250, s1: 420, side: 'L', offset: 3 },
  ],
};

export const TRACKS: Record<string, TrackDef> = {
  [CRIMSON_BAY.id]: CRIMSON_BAY,
};
