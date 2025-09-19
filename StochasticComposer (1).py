
# --- Stochastic Composer (Single-file UI + Engine + MIDI + MusicXML + WAV) ---
# Purely stochastic generation driven by style heuristics.
# Requires only the Python standard library.

import random
import struct
import math
import wave
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# =========================
# Core Music Data Structures
# =========================

@dataclass
class NoteEvent:
    start: int      # in ticks
    duration: int   # in ticks
    pitch: int      # MIDI note number 0-127
    velocity: int   # 1-127
    channel: int = 0

# =========================
# Utility Functions
# =========================

def vlf_write(value: int) -> bytes:
    """Write a number as MIDI variable-length quantity."""
    buffer = value & 0x7F
    out = bytearray([buffer])
    value >>= 7
    while value:
        buffer = (value & 0x7F) | 0x80
        out.insert(0, buffer)
        value >>= 7
    return bytes(out)

def midi_tempo_from_bpm(bpm: int) -> int:
    """Microseconds per quarter note for MIDI tempo meta event."""
    return int(60_000_000 / max(1, bpm))

def clamp(n, lo, hi):
    return max(lo, min(hi, n))

# =========================
# Scale Builder
# =========================

NOTE_NAMES_SHARP = ["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"]
NOTE_NAME_TO_SEMITONE = {n:i for i,n in enumerate(NOTE_NAMES_SHARP)}

MAJOR_STEPS = [2,2,1,2,2,2,1]       # Ionian
NAT_MINOR_STEPS = [2,1,2,2,1,2,2]   # Aeolian

def build_scale(key_root: str, mode: str, octaves=(4,5)) -> List[int]:
    """Return MIDI pitches for the diatonic scale over the given octave range (C-based before transpose)."""
    steps = MAJOR_STEPS if mode.lower() == "major" else NAT_MINOR_STEPS
    pcs = [0]
    for step in steps[:-1]:
        pcs.append(pcs[-1] + step)
    notes = []
    for oct_ in range(octaves[0], octaves[1]+1):
        base = 12 * (oct_ + 1)  # C4=60
        for deg in pcs:
            notes.append(base + deg)
    return notes

def chromatic_range(octaves=(4,5)) -> List[int]:
    notes = []
    for oct_ in range(octaves[0], octaves[1]+1):
        base = 12 * (oct_ + 1)
        for i in range(12):
            notes.append(base + i)
    return notes

def transpose_to_key(pitches: List[int], key_root: str) -> List[int]:
    root_pc = NOTE_NAME_TO_SEMITONE[key_root]
    delta = root_pc
    return [p + delta for p in pitches]

# =========================
# Style Definitions
# =========================

STYLE_LIBRARY: Dict[str, Dict] = {
    "Classical": {"tempo": (88, 120), "program": 0, "rhythms": [480,240,240,120,120,360],
                  "interval_weights": {0:3,2:8,-2:8,4:2,-4:2,5:1,-5:1,7:1,-7:1}, "chord_tone_bias": 0.65},
    "Jazz": {"tempo": (120, 180), "program": 4, "rhythms": [240,240,240,120,360,480],
             "interval_weights": {0:2,1:6,-1:6,2:6,-2:6,3:2,-3:2,4:3,-4:3}, "chord_tone_bias": 0.5, "default_swing": 0.16},
    "Waltz": {"tempo": (84, 108), "program": 0, "rhythms": [320,160,160,160,480],
              "interval_weights": {0:2,2:7,-2:7,4:3,-4:3,5:1,-5:1}, "chord_tone_bias": 0.6, "meter": (3,4)},
    "Blues": {"tempo": (90, 130), "program": 26, "rhythms": [360,120,240,240,480],
              "interval_weights": {0:2,1:3,-1:3,2:6,-2:6,3:4,-3:2,4:2,-4:2,6:1,-6:1}, "chord_tone_bias": 0.55, "default_swing": 0.12},
    "Rock": {"tempo": (100, 140), "program": 30, "rhythms": [240,240,120,120,480],
             "interval_weights": {0:2,2:7,-2:7,4:3,-4:3,7:2,-7:2}, "chord_tone_bias": 0.45},
    "Pop": {"tempo": (96, 128), "program": 0, "rhythms": [240,240,240,120,480],
            "interval_weights": {0:3,2:8,-2:8,4:2,-4:2,5:2,-5:2}, "chord_tone_bias": 0.5},
    "HipHop": {"tempo": (78, 96), "program": 1, "rhythms": [240,120,120,480,360],
               "interval_weights": {0:4,1:6,-1:6,2:4,-2:4,3:2,-3:2}, "chord_tone_bias": 0.4, "default_swing": 0.14},
    "Techno": {"tempo": (120, 140), "program": 81, "rhythms": [120,120,120,120,240,480],
               "interval_weights": {0:2,1:5,-1:5,2:5,-2:5,3:2,-3:2}, "chord_tone_bias": 0.35},
    "Ambient": {"tempo": (60, 84), "program": 89, "rhythms": [480,720,960],
                "interval_weights": {0:5,2:4,-2:4,5:2,-5:2,7:1,-7:1}, "chord_tone_bias": 0.7},
    "FolkGuitar": {"tempo": (92, 120), "program": 24, "rhythms": [240,240,360,120,480],
                   "interval_weights": {0:2,2:7,-2:7,3:2,-3:2,4:2,-4:2,5:2,-5:2}, "chord_tone_bias": 0.55},
    "MiddleEast": {"tempo": (90, 120), "program": 104, "rhythms": [240,240,120,120,360],
                   "interval_weights": {0:2,1:6,-1:6,2:5,-2:5,3:2,-3:2,6:1,-6:1}, "chord_tone_bias": 0.5},
    "Carnatic": {"tempo": (84, 112), "program": 104, "rhythms": [120,120,120,240,360],
                 "interval_weights": {0:2,1:5,-1:5,2:6,-2:6,3:3,-3:3,4:2,-4:2}, "chord_tone_bias": 0.6},
    "Latin": {"tempo": (96, 132), "program": 0, "rhythms": [240,120,120,360,480],
              "interval_weights": {0:2,2:6,-2:6,3:3,-3:3,4:2,-4:2,5:2,-5:2}, "chord_tone_bias": 0.5},
    "Reggae": {"tempo": (70, 88), "program": 2, "rhythms": [240,240,480,360],
               "interval_weights": {0:3,2:6,-2:6,4:3,-4:3,7:2,-7:2}, "chord_tone_bias": 0.5},
    "LoFi": {"tempo": (60, 84), "program": 5, "rhythms": [240,240,120,360,480],
             "interval_weights": {0:4,1:5,-1:5,2:5,-2:5,3:2,-3:2}, "chord_tone_bias": 0.55, "default_swing": 0.1},
    "Baroque": {"tempo": (88, 120), "program": 7, "rhythms": [240,240,120,120,360,480],
                "interval_weights": {0:2,2:8,-2:8,4:4,-4:4,5:3,-5:3}, "chord_tone_bias": 0.7},
    "Cinematic": {"tempo": (72, 110), "program": 48, "rhythms": [480,720,960,240],
                  "interval_weights": {0:4,2:5,-2:5,5:3,-5:3,7:2,-7:2}, "chord_tone_bias": 0.65},
    "Chiptune": {"tempo": (120, 160), "program": 80, "rhythms": [120,120,120,240,480],
                 "interval_weights": {0:2,2:6,-2:6,4:3,-4:3,7:2,-7:2}, "chord_tone_bias": 0.45},
}

# =========================
# Stochastic Engine
# =========================

class StochasticComposer:
    def __init__(self, tpq=480):
        self.tpq = tpq

    def _choose_tempo(self, style_name: str, user_bpm: Optional[int]) -> int:
        if user_bpm:
            return clamp(int(user_bpm), 40, 220)
        lo, hi = STYLE_LIBRARY[style_name]["tempo"]
        return random.randint(lo, hi)

    def _interval_from_weights(self, weights: Dict[int,int]) -> int:
        choices = []
        for interval, w in weights.items():
            choices.extend([interval] * int(max(1, w)))
        return random.choice(choices)

    def _build_pitch_pool(self, key_root: str, mode: str, chromatic: bool, octaves=(4,5)) -> List[int]:
        if chromatic:
            base = chromatic_range(octaves)
            return transpose_to_key(base, key_root)
        else:
            base = build_scale(key_root, mode, octaves)
            return transpose_to_key(base, key_root)

    def _is_chord_tone(self, midi_pitch: int, key_root: str, mode: str) -> bool:
        root_pc = NOTE_NAME_TO_SEMITONE[key_root]
        pc = midi_pitch % 12
        if mode.lower() == "major":
            chord_pcs = {(root_pc + 0) % 12, (root_pc + 4) % 12, (root_pc + 7) % 12}
        else:
            chord_pcs = {(root_pc + 0) % 12, (root_pc + 3) % 12, (root_pc + 7) % 12}
        return pc in chord_pcs

    def _nearest_in_pool(self, target: int, pool: List[int]) -> int:
        return min(pool, key=lambda p: abs(p - target))

    def _pick_next_pitch(self, current: int, pool: List[int], style: Dict, chord_bias: float, key_root:str, mode:str) -> int:
        interval = self._interval_from_weights(style["interval_weights"])
        candidate = current + interval
        candidate = self._nearest_in_pool(candidate, pool)
        if random.random() < chord_bias:
            candidates = sorted(pool, key=lambda p: abs(p - candidate))[:5]
            chordies = [p for p in candidates if self._is_chord_tone(p, key_root, mode)]
            if chordies:
                candidate = random.choice(chordies)
        return clamp(candidate, 36, 90)

    def _choose_rhythm_sequence(self, style: Dict, total_ticks: int) -> List[int]:
        out, remaining = [], total_ticks
        cells = style["rhythms"]
        while remaining > 0:
            cell = random.choice(cells)
            if cell > remaining:
                smaller = [c for c in cells if c <= remaining]
                if not smaller:
                    out.append(remaining); remaining = 0
                else:
                    sel = random.choice(smaller); out.append(sel); remaining -= sel
            else:
                out.append(cell); remaining -= cell
        return out

    def _apply_swing(self, notes: List[NoteEvent], meta: Dict, swing: float) -> List[NoteEvent]:
        """Delay off-beat eighths by swing*ticks (0..0.3)."""
        if swing <= 0: return notes
        tpq = meta["tpq"]
        beats, beat_unit = meta["meter"]
        beat_ticks = int(tpq * (4/beat_unit))  # quarter-note in ticks
        eighth = tpq // 2
        delay = int(eighth * swing)
        out = []
        for n in notes:
            pos_in_beat = n.start % beat_ticks
            # If this note begins on an off-eighth (i.e., odd multiple of eighth but not on beat)
            if (pos_in_beat % eighth == 0) and (pos_in_beat % beat_ticks != 0) and ((pos_in_beat // eighth) % 2 == 1):
                new_start = n.start + delay
                new_dur = max(1, n.duration - delay)
                out.append(NoteEvent(start=new_start, duration=new_dur, pitch=n.pitch, velocity=n.velocity, channel=n.channel))
            else:
                out.append(n)
        return out

    def _voice_bass(self, key_root:str, mode:str, pitch_mode:str, bars:int, style:str, bpm:int, seed=None) -> Tuple[List[NoteEvent], Dict]:
        random.seed(seed)
        style_def = STYLE_LIBRARY[style]
        pool = self._build_pitch_pool(key_root, mode, pitch_mode.lower()=="chromatic", (2,3))
        meter = style_def.get("meter",(4,4))
        beats, beat_unit = meter
        ticks_per_bar = int(self.tpq * (4/beat_unit) * beats)
        chord_bias = 0.85
        current = self._nearest_in_pool(48, pool)
        t=0
        notes=[]
        for _ in range(bars):
            # quarters/halves bias
            rhythm_seq = []
            remaining = ticks_per_bar
            while remaining>0:
                cell = random.choice([self.tpq, self.tpq*2, self.tpq//2, self.tpq])
                if cell>remaining: cell = remaining
                rhythm_seq.append(cell); remaining-=cell
            for dur in rhythm_seq:
                pitch = self._pick_next_pitch(current, pool, style_def, chord_bias, key_root, mode)
                vel = random.randint(60, 95)
                notes.append(NoteEvent(start=t, duration=dur, pitch=pitch-12, velocity=vel, channel=1))
                current = pitch
                t+=dur
        meta={"bpm":bpm,"program":32,"meter":meter,"tpq":self.tpq,"style":style,"key":key_root,"mode":mode,"pitch_mode":pitch_mode}
        return notes, meta

    def _triad_for_degree(self, degree:int, key_root:str, mode:str) -> List[int]:
        # Build diatonic triad on degree (0..6) relative to key
        scale = transpose_to_key(build_scale(key_root, mode, (3,6)), key_root)
        # take degrees within one octave
        base = [p for p in scale if 60 <= p <= 72]
        if len(base) < 7:
            # fallback to C-based
            base = [60,62,64,65,67,69,71]
        root = base[degree%7]
        third = base[(degree+2)%7]
        fifth = base[(degree+4)%7]
        return [root, third, fifth]

    def _voice_chords(self, key_root:str, mode:str, pitch_mode:str, bars:int, style:str, bpm:int, seed=None) -> Tuple[List[NoteEvent], Dict]:
        random.seed(seed)
        style_def = STYLE_LIBRARY[style]
        meter = style_def.get("meter",(4,4))
        beats, beat_unit = meter
        ticks_per_bar = int(self.tpq * (4/beat_unit) * beats)
        t=0; notes=[]
        # Simple stochastic progression using I, IV, V, vi, ii
        degrees = [0,3,4,5,1]  # I, IV, V, vi, ii (0-based)
        weights = [5,3,4,2,2]
        for _ in range(bars):
            deg = random.choices(degrees, weights=weights, k=1)[0]
            triad = self._triad_for_degree(deg, key_root, mode)
            # choose rhythm: whole or two halves
            if random.random()<0.6:
                dur = ticks_per_bar
                for p in triad:
                    notes.append(NoteEvent(start=t, duration=dur, pitch=clamp(p,40,84), velocity=random.randint(50,90), channel=2))
                t += dur
            else:
                half = ticks_per_bar//2
                for k in range(2):
                    for p in triad:
                        notes.append(NoteEvent(start=t+k*half, duration=half, pitch=clamp(p+(0 if k==0 else 12),40,88), velocity=random.randint(50,90), channel=2))
                t += ticks_per_bar
        meta={"bpm":bpm,"program":48,"meter":meter,"tpq":self.tpq,"style":style,"key":key_root,"mode":mode,"pitch_mode":pitch_mode}
        return notes, meta

    def generate(self,
                 key_root: str = "C",
                 mode: str = "Major",
                 pitch_mode: str = "Diatonic",
                 style_name: str = "Classical",
                 bars: int = 8,
                 bpm: Optional[int] = None,
                 seed: Optional[int] = None,
                 swing: float = 0.0,
                 voices: str = "Solo") -> Tuple[List[NoteEvent], Dict]:
        """voices: 'Solo', 'Melody+Bass', 'Melody+Chords', 'Trio'"""
        if seed is not None:
            random.seed(seed)

        style = STYLE_LIBRARY[style_name]
        bpm_final = self._choose_tempo(style_name, bpm)
        meter = style.get("meter", (4,4))
        beats_per_bar, beat_unit = meter
        ticks_per_bar = int(self.tpq * (4/beat_unit) * beats_per_bar)

        chromatic = (pitch_mode.lower() == "chromatic")
        pool = self._build_pitch_pool(key_root, mode, chromatic, (3,6))

        current_pitch = self._nearest_in_pool(60, pool)

        notes: List[NoteEvent] = []
        t = 0
        chord_bias = style.get("chord_tone_bias", 0.5)

        for _ in range(bars):
            rhythm_seq = self._choose_rhythm_sequence(style, ticks_per_bar)
            for dur in rhythm_seq:
                pitch = self._pick_next_pitch(current_pitch, pool, style, chord_bias, key_root, mode)
                vel = random.randint(64, 110)
                notes.append(NoteEvent(start=t, duration=dur, pitch=pitch, velocity=vel, channel=0))
                t += dur
                current_pitch = pitch

        meta = {"bpm": bpm_final, "program": style["program"], "meter": meter, "tpq": self.tpq,
                "style": style_name, "key": key_root, "mode": mode, "pitch_mode": pitch_mode}

        # Additional voices
        all_notes = list(notes)
        if voices in ("Melody+Bass","Trio"):
            bass_notes, _ = self._voice_bass(key_root, mode, pitch_mode, bars, style_name, bpm_final, seed=None if seed is None else seed+1)
            all_notes.extend(bass_notes)
        if voices in ("Melody+Chords","Trio"):
            chord_notes, _ = self._voice_chords(key_root, mode, pitch_mode, bars, style_name, bpm_final, seed=None if seed is None else seed+2)
            all_notes.extend(chord_notes)

        # Determine swing default if not set explicitly
        if swing == -1.0:
            swing = style.get("default_swing", 0.0)
        # Apply swing
        all_notes = self._apply_swing(all_notes, meta, swing)

        return all_notes, meta

# =========================
# MIDI Writer (single track, multiple channels)
# =========================

def write_midi(filename: str, notes: List[NoteEvent], meta: Dict):
    tpq = meta["tpq"]
    bpm = meta["bpm"]
    program = meta["program"]

    # Build events list (tempo + one program per channel 0..2)
    events_list = []
    for n in notes:
        events_list.append((n.start, "on", n.pitch, clamp(n.velocity,1,127), n.channel))
        events_list.append((n.start + n.duration, "off", n.pitch, 0, n.channel))

    events_list.sort(key=lambda x: (x[0], 0 if x[1]=="off" else 1))

    events = bytearray()

    # Tempo meta
    tempo = midi_tempo_from_bpm(bpm)
    events += vlf_write(0) + bytes([0xFF, 0x51, 0x03]) + struct.pack(">I", tempo)[1:]

    # Program change per channel used
    channels_used = sorted({ch for _,_,_,_,ch in events_list})
    for ch in channels_used:
        prog = program if ch==0 else (32 if ch==1 else 48)
        events += vlf_write(0) + bytes([0xC0 | (ch & 0x0F), clamp(prog,0,127)])

    last_time = 0
    for t, kind, pitch, vel, ch in events_list:
        delta = t - last_time
        if delta < 0: delta = 0
        events += vlf_write(delta)
        if kind == "on":
            events += bytes([0x90 | (ch & 0x0F), clamp(pitch,0,127), vel])
        else:
            events += bytes([0x80 | (ch & 0x0F), clamp(pitch,0,127), 0])
        last_time = t

    events += vlf_write(0) + bytes([0xFF, 0x2F, 0x00])
    header = b"MThd" + struct.pack(">IHHH", 6, 1, 1, tpq)
    track = b"MTrk" + struct.pack(">I", len(events)) + events

    with open(filename, "wb") as f:
        f.write(header + track)
    return filename

# =========================
# MusicXML Writer (Simple Partwise)
# =========================

def pitch_to_xml(pitch: int) -> Tuple[str, int, int]:
    steps = ["C","C","D","E","E","F","F","G","A","A","B","B"]
    alters = [0,1,0,-1,0,0,1,0,-1,0,-1,0]
    step = steps[pitch % 12]
    alter = alters[pitch % 12]
    octave = pitch//12 - 1
    return step, alter, octave

def write_musicxml(filename: str, notes: List[NoteEvent], meta: Dict):
    tpq = meta["tpq"]
    bpm = meta["bpm"]
    beats, beat_type = meta["meter"]
    key_root = meta["key"]
    mode = meta["mode"]

    divisions = tpq
    key_to_fifths = {"C":0,"G":1,"D":2,"A":3,"E":4,"B":5,"F#":6,"C#":7,
                     "F":-1,"Bb":-2,"Eb":-3,"Ab":-4,"Db":-5,"Gb":-6,"Cb":-7}
    fifths = key_to_fifths.get(key_root, 0)
    mode_xml = "major" if mode.lower()=="major" else "minor"

    ticks_per_bar = int(tpq * (4/beat_type) * beats)

    events = []
    cursor = 0
    for n in sorted(notes, key=lambda x: x.start):
        if n.start > cursor:
            events.append(("rest", n.start - cursor, None))
            cursor = n.start
        events.append(("note", n.duration, n.pitch))
        cursor += n.duration

    measures = []
    mtime = 0
    current = []
    for kind, dur, pitch in events:
        remain = dur
        while remain > 0:
            space = ticks_per_bar - (mtime % ticks_per_bar)
            take = min(space, remain)
            current.append((kind, take, pitch))
            mtime += take
            remain -= take
            if (mtime % ticks_per_bar) == 0:
                measures.append(current)
                current = []
    if current:
        measures.append(current)

    out = []
    out.append('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
    out.append('<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">')
    out.append('<score-partwise version="3.1">')
    out.append('  <part-list><score-part id="P1"><part-name>Stochastic</part-name></score-part></part-list>')
    out.append('  <part id="P1">')

    for i, measure in enumerate(measures, start=1):
        out.append(f'    <measure number="{i}">')
        if i == 1:
            out.append('      <attributes>')
            out.append(f'        <divisions>{divisions}</divisions>')
            out.append('        <key>')
            out.append(f'          <fifths>{fifths}</fifths>')
            out.append(f'          <mode>{mode_xml}</mode>')
            out.append('        </key>')
            out.append('        <time>')
            out.append(f'          <beats>{beats}</beats>')
            out.append(f'          <beat-type>{beat_type}</beat-type>')
            out.append('        </time>')
            out.append('        <clef><sign>G</sign><line>2</line></clef>')
            out.append('      </attributes>')
            out.append('      <direction placement="above">')
            out.append('        <direction-type><metronome><beat-unit>quarter</beat-unit><per-minute>{}</per-minute></metronome></direction-type>'.format(bpm))
            out.append('      </direction>')

        for kind, dur, pitch in measure:
            if dur <= 0: 
                continue
            if kind == "rest":
                out.append('      <note>')
                out.append('        <rest/>')
                out.append(f'        <duration>{dur}</duration>')
                out.append('        <voice>1</voice>')
                out.append('      </note>')
            else:
                step, alter, octave = pitch_to_xml(pitch)
                out.append('      <note>')
                out.append('        <pitch>')
                out.append(f'          <step>{step}</step>')
                if alter != 0:
                    out.append(f'          <alter>{alter}</alter>')
                out.append(f'          <octave>{octave}</octave>')
                out.append('        </pitch>')
                out.append(f'        <duration>{dur}</duration>')
                out.append('        <voice>1</voice>')
                out.append('      </note>')
        out.append('    </measure>')

    out.append('  </part>')
    out.append('</score-partwise>')

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    return filename

# =========================
# Simple WAV Renderer (no deps) - basic polysynth
# =========================

def midi_pitch_to_freq(midi: int) -> float:
    return 440.0 * (2 ** ((midi - 69) / 12.0))

def render_wav(filename: str, notes: List[NoteEvent], meta: Dict, sr: int = 44100):
    """Render to WAV using a simple sine+square hybrid and linear envelope. Mono."""
    tpq = meta["tpq"]
    bpm = meta["bpm"]
    sec_per_tick = (60.0 / bpm) / tpq

    # Determine total length
    total_ticks = 0
    for n in notes:
        total_ticks = max(total_ticks, n.start + n.duration)
    total_samples = int((total_ticks * sec_per_tick) * sr) + sr//2

    buf = [0.0] * total_samples

    for n in notes:
        freq = midi_pitch_to_freq(n.pitch)
        amp = n.velocity / 127.0 * 0.25  # global attenuation
        start_sample = int(n.start * sec_per_tick * sr)
        dur_samples = max(1, int(n.duration * sec_per_tick * sr))
        # ADSR-ish: quick attack, decay to 70%, release
        a = max(1, int(0.01 * sr))
        r = max(1, int(0.05 * sr))
        for i in range(dur_samples):
            t = (start_sample + i)
            if t >= total_samples: break
            env = 1.0
            if i < a:
                env = i / a
            elif i > dur_samples - r:
                env = max(0.0, (dur_samples - i) / r)
            # hybrid waveform
            val = math.sin(2*math.pi*freq*(t/sr)) * 0.7 + (1 if (int(2*freq*(t/sr)) % 2 == 0) else -1) * 0.3
            buf[t] += amp * env * val

    # Normalize to 16-bit
    peak = max(1e-6, max(abs(x) for x in buf))
    norm = 0.95 / peak
    with wave.open(filename, "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        for x in buf:
            w.writeframesraw(struct.pack("<h", int(clamp(int(x*norm*32767), -32768, 32767))))
    return filename

# =========================
# Tkinter UI
# =========================

def run_ui():
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog

    KEYS = ['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B']
    SCALES = ['Major','Minor']
    PITCH_MODES = ['Diatonic','Chromatic']
    STYLES = list(STYLE_LIBRARY.keys())
    VOICES = ['Solo','Melody+Bass','Melody+Chords','Trio']

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("Purely Stochastic Composer")
            self.geometry("880x600")

            frm = ttk.Frame(self); frm.pack(fill='both', expand=True, padx=12, pady=12)

            row=0
            ttk.Label(frm, text="Key").grid(row=row, column=0, sticky='w')
            self.key_var = tk.StringVar(value='C')
            ttk.Combobox(frm, textvariable=self.key_var, values=KEYS, width=6).grid(row=row, column=1, sticky='w')

            ttk.Label(frm, text="Scale").grid(row=row, column=2, sticky='w', padx=(16,0))
            self.scale_var = tk.StringVar(value='Major')
            ttk.Combobox(frm, textvariable=self.scale_var, values=SCALES, width=8).grid(row=row, column=3, sticky='w')

            ttk.Label(frm, text="Pitch Mode").grid(row=row, column=4, sticky='w', padx=(16,0))
            self.pitch_mode_var = tk.StringVar(value='Diatonic')
            ttk.Combobox(frm, textvariable=self.pitch_mode_var, values=PITCH_MODES, width=10).grid(row=row, column=5, sticky='w')

            row+=1
            ttk.Label(frm, text="Style").grid(row=row, column=0, sticky='w', pady=(8,0))
            self.style_var = tk.StringVar(value=STYLES[0])
            ttk.Combobox(frm, textvariable=self.style_var, values=STYLES, width=18).grid(row=row, column=1, sticky='w', pady=(8,0))

            ttk.Label(frm, text="Bars").grid(row=row, column=2, sticky='w', padx=(16,0), pady=(8,0))
            self.bars_var = tk.IntVar(value=8)
            ttk.Spinbox(frm, from_=4, to=128, textvariable=self.bars_var, width=6).grid(row=row, column=3, sticky='w', pady=(8,0))

            ttk.Label(frm, text="Tempo (optional)").grid(row=row, column=4, sticky='w', padx=(16,0), pady=(8,0))
            self.tempo_var = tk.StringVar(value="")
            ttk.Entry(frm, textvariable=self.tempo_var, width=8).grid(row=row, column=5, sticky='w', pady=(8,0))

            row+=1
            ttk.Label(frm, text="Random Seed (optional)").grid(row=row, column=0, sticky='w', pady=(8,0))
            self.seed_var = tk.StringVar(value="")
            ttk.Entry(frm, textvariable=self.seed_var, width=12).grid(row=row, column=1, sticky='w', pady=(8,0))

            ttk.Label(frm, text="Swing % (0–30)").grid(row=row, column=2, sticky='w', padx=(16,0), pady=(8,0))
            self.swing_var = tk.IntVar(value=0)
            ttk.Spinbox(frm, from_=0, to=30, textvariable=self.swing_var, width=6).grid(row=row, column=3, sticky='w', pady=(8,0))

            ttk.Label(frm, text="Voices").grid(row=row, column=4, sticky='w', padx=(16,0), pady=(8,0))
            self.voices_var = tk.StringVar(value='Solo')
            ttk.Combobox(frm, textvariable=self.voices_var, values=VOICES, width=14).grid(row=row, column=5, sticky='w', pady=(8,0))

            row+=1
            btn_frm = ttk.Frame(frm); btn_frm.grid(row=row, column=0, columnspan=6, pady=(12,6), sticky='w')
            ttk.Button(btn_frm, text="Generate", command=self.on_generate).pack(side='left', padx=4)
            ttk.Button(btn_frm, text="Export MIDI", command=self.on_export_midi).pack(side='left', padx=4)
            ttk.Button(btn_frm, text="Export MusicXML", command=self.on_export_xml).pack(side='left', padx=4)
            ttk.Button(btn_frm, text="Export WAV", command=self.on_export_wav).pack(side='left', padx=4)

            row+=1
            ttk.Label(frm, text="Log").grid(row=row, column=0, sticky='w')
            row+=1
            self.log = tk.Text(frm, height=16)
            self.log.grid(row=row, column=0, columnspan=6, sticky='nsew', pady=(4,0))
            frm.rowconfigure(row, weight=1); frm.columnconfigure(5, weight=1)

            self.last_notes = None; self.last_meta = None

        def _parse_int(self, var):
            try:
                return int(var.get()) if str(var.get()).strip() else None
            except: return None

        def on_generate(self):
            key = self.key_var.get()
            scale = self.scale_var.get()
            pitch_mode = self.pitch_mode_var.get()
            style = self.style_var.get()
            bars = int(self.bars_var.get())
            tempo = self._parse_int(self.tempo_var)
            seed = self._parse_int(self.seed_var)
            swing_pct = int(self.swing_var.get())
            voices = self.voices_var.get()

            comp = StochasticComposer()
            # swing = -1.0 uses style default; else use provided percent
            swing = (swing_pct/100.0) if swing_pct>0 else -1.0
            notes, meta = comp.generate(key_root=key, mode=scale, pitch_mode=pitch_mode,
                                        style_name=style, bars=bars, bpm=tempo, seed=seed,
                                        swing=swing, voices=voices)
            self.last_notes, self.last_meta = notes, meta

            self.log.delete("1.0", "end")
            self.log.insert("end", f"Generated {len(notes)} events. Style={style}, Key={key} {scale}, PitchMode={pitch_mode}\n")
            self.log.insert("end", f"BPM={meta['bpm']}, Meter={meta['meter']}, Voices={voices}, Swing={(swing if swing>=0 else meta.get('style',''))}\n")
            for n in notes[:12]:
                self.log.insert("end", f"t={n.start} dur={n.duration} pitch={n.pitch} vel={n.velocity} ch={n.channel}\n")

        def on_export_midi(self):
            if not self.last_notes:
                from tkinter import messagebox; messagebox.showinfo("Info","Please Generate first."); return
            from tkinter import filedialog
            fn = filedialog.asksaveasfilename(defaultextension=".mid", filetypes=[("MIDI","*.mid")])
            if not fn: return
            path = write_midi(fn, self.last_notes, self.last_meta)
            from tkinter import messagebox; messagebox.showinfo("Saved", f"MIDI saved to {path}")

        def on_export_xml(self):
            if not self.last_notes:
                from tkinter import messagebox; messagebox.showinfo("Info","Please Generate first."); return
            from tkinter import filedialog
            fn = filedialog.asksaveasfilename(defaultextension=".musicxml", filetypes=[("MusicXML","*.musicxml")])
            if not fn: return
            path = write_musicxml(fn, self.last_notes, self.last_meta)
            from tkinter import messagebox; messagebox.showinfo("Saved", f"MusicXML saved to {path}")

        def on_export_wav(self):
            if not self.last_notes:
                from tkinter import messagebox; messagebox.showinfo("Info","Please Generate first."); return
            from tkinter import filedialog
            fn = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV","*.wav")])
            if not fn: return
            path = render_wav(fn, self.last_notes, self.last_meta)
            from tkinter import messagebox; messagebox.showinfo("Saved", f"WAV saved to {path}")

    app = App()
    app.mainloop()

if __name__ == "__main__":
    run_ui()
