import os
import re
import io
import shutil
import tempfile
import numpy as np
import librosa
import yt_dlp
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
from flask_cors import CORS
from hmmlearn import hmm

app = Flask(__name__)
CORS(app)

AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio_cache')
os.makedirs(AUDIO_DIR, exist_ok=True)

# Krumhansl-Schmuckler key profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                           2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']

# ── Chord templates: major, minor, 7, m7, dim, aug, sus2, sus4 ──
def _build_chord_templates():
    # intervals relative to root (in semitones)
    CHORD_TYPES = {
        '':     [0, 4, 7],       # major
        'm':    [0, 3, 7],       # minor
        '7':    [0, 4, 7, 10],   # dominant 7th
        'm7':   [0, 3, 7, 10],   # minor 7th
        '7M':   [0, 4, 7, 11],   # major 7th
        'dim':  [0, 3, 6],       # diminished
        'aug':  [0, 4, 8],       # augmented
        'sus2': [0, 2, 7],       # suspended 2nd
        'sus4': [0, 5, 7],       # suspended 4th
    }
    templates = {}  # name -> normalized 12-d vector
    for i, root in enumerate(NOTE_NAMES):
        for suffix, intervals in CHORD_TYPES.items():
            vec = np.zeros(12)
            for iv in intervals:
                vec[(i + iv) % 12] = 1.0
            templates[f"{root}{suffix}"] = vec / np.linalg.norm(vec)
    return templates

CHORD_TEMPLATES = _build_chord_templates()
CHORD_NAMES = list(CHORD_TEMPLATES.keys())      # deterministic order
CHORD_MATRIX = np.array([CHORD_TEMPLATES[n] for n in CHORD_NAMES])  # (N_chords, 12)
N_CHORDS = len(CHORD_NAMES)

# Simplicity prior: triads are far more common than complex chords.
# Complex chords must score significantly higher to win over a simple triad.
CHORD_PRIOR = np.array([
    {
        '': 1.0,        # major triad — no penalty
        'm': 1.0,       # minor triad — no penalty
        '7': 0.82,      # dominant 7th — slight penalty
        'm7': 0.80,     # minor 7th
        '7M': 0.78,     # major 7th
        'sus2': 0.75,   # suspended chords are rarer
        'sus4': 0.75,
        'dim': 0.72,    # diminished
        'aug': 0.72,    # augmented
    }[suffix]
    for root in NOTE_NAMES
    for suffix in ['', 'm', '7', 'm7', '7M', 'dim', 'aug', 'sus2', 'sus4']
])


def _extract_root(chord_name: str) -> str:
    """Extract root note from chord name like 'C#m7' -> 'C#'."""
    if len(chord_name) >= 2 and chord_name[1] == '#':
        return chord_name[:2]
    return chord_name[0]


def _build_hmm_model() -> hmm.CategoricalHMM:
    """
    Build a Hidden Markov Model for chord sequence decoding.
    - Transition matrix: high self-transition (chords sustain), small uniform
      probability for switching, with bonus for musically common transitions.
    - Emission is handled externally (we use the Viterbi path on score indices).
    """
    SELF_PROB = 0.70  # probability of staying on same chord
    model = hmm.CategoricalHMM(n_components=N_CHORDS, n_iter=0, init_params="")

    # Uniform start
    model.startprob_ = np.ones(N_CHORDS) / N_CHORDS

    # Transition matrix with musical priors
    trans = np.full((N_CHORDS, N_CHORDS), (1.0 - SELF_PROB) / (N_CHORDS - 1))
    np.fill_diagonal(trans, SELF_PROB)

    # Boost common harmonic movements (4th, 5th, relative minor/major)
    COMMON_INTERVALS = [5, 7, 3, 4, 9, 8, 2]  # P4, P5, m3, M3, M6, m6, M2
    for i, name_i in enumerate(CHORD_NAMES):
        root_i_str = _extract_root(name_i)
        if root_i_str not in NOTE_NAMES:
            continue
        root_i = NOTE_NAMES.index(root_i_str)
        for j, name_j in enumerate(CHORD_NAMES):
            if i == j:
                continue
            root_j_str = _extract_root(name_j)
            if root_j_str not in NOTE_NAMES:
                continue
            root_j = NOTE_NAMES.index(root_j_str)
            interval = (root_j - root_i) % 12
            if interval in COMMON_INTERVALS:
                trans[i, j] *= 2.5  # boost common transitions

    # Re-normalize rows
    trans /= trans.sum(axis=1, keepdims=True)
    model.transmat_ = trans

    # Emission probs are set dynamically per song (dummy here)
    model.emissionprob_ = np.eye(N_CHORDS)
    model.n_features = N_CHORDS
    return model

_HMM_MODEL = _build_hmm_model()


def detect_chords(y, sr, hop_length=512) -> list[dict]:
    """
    Detect chords using NNLS chroma + beat-synchronous segmentation + HMM decoding.
    """
    # ── 1. Feature extraction ──
    y_harmonic = librosa.effects.harmonic(y, margin=4)

    # NNLS chroma: fits harmonics precisely, much cleaner than raw CQT
    chroma_nnls = librosa.feature.chroma_cqt(
        y=y_harmonic, sr=sr, hop_length=hop_length,
        bins_per_octave=36, n_octaves=6
    )
    chroma_cens = librosa.feature.chroma_cens(
        y=y_harmonic, sr=sr, hop_length=hop_length
    )
    min_frames = min(chroma_nnls.shape[1], chroma_cens.shape[1])
    chroma = chroma_nnls[:, :min_frames] * 0.65 + chroma_cens[:, :min_frames] * 0.35

    # Light median smoothing (size=3: removes only single-frame spikes)
    from scipy.ndimage import median_filter
    chroma = median_filter(chroma, size=(1, 3))

    # ── 2. Beat-synchronous + sub-beat segmentation ──
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    _, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length
    )

    # Create boundaries at each beat AND at each half-beat
    all_boundaries = [0]
    for i in range(len(beat_frames)):
        if i > 0:
            mid = (beat_frames[i - 1] + beat_frames[i]) // 2
            all_boundaries.append(int(mid))
        all_boundaries.append(int(beat_frames[i]))
    all_boundaries.append(chroma.shape[1])
    all_boundaries = sorted(set(all_boundaries))

    # ── 3. Score each segment against all chord templates ──
    segment_times = []
    score_matrix = []  # (n_segments, N_CHORDS)

    for i in range(len(all_boundaries) - 1):
        sf = all_boundaries[i]
        ef = all_boundaries[i + 1]
        if ef - sf < 2:
            continue

        seg = chroma[:, sf:ef].mean(axis=1)
        norm = np.linalg.norm(seg)
        if norm < 1e-6:
            continue
        seg = seg / norm

        # Dot product against all templates at once (vectorized)
        scores = (CHORD_MATRIX @ seg) * CHORD_PRIOR  # apply simplicity bias
        score_matrix.append(scores)
        segment_times.append(round(sf * hop_length / sr, 2))

    if not score_matrix:
        return []

    score_matrix = np.array(score_matrix)  # (n_segments, N_CHORDS)

    # ── 4. HMM Viterbi decoding for optimal chord sequence ──
    # Convert scores to pseudo-probabilities via softmax
    temperature = 0.15  # lower = more peaky = more confident
    exp_scores = np.exp(score_matrix / temperature)
    emission_probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    # Build per-song emission matrix and decode
    model = _HMM_MODEL
    # Use log-likelihood Viterbi via hmmlearn's decode
    # We need to pass observations as indices, but we have continuous emissions.
    # Workaround: use GaussianHMM-style or manual Viterbi.
    best_path = _viterbi_decode(model.startprob_, model.transmat_, emission_probs)

    # ── 5. Convert to output format, emit only on changes ──
    chords = []
    prev_chord = None
    for idx, seg_idx in enumerate(best_path):
        chord = CHORD_NAMES[seg_idx]
        if chord != prev_chord:
            chords.append({"time": segment_times[idx], "chord": chord})
            prev_chord = chord

    return chords


def _viterbi_decode(startprob, transmat, emission_probs):
    """
    Standard Viterbi algorithm for HMM decoding with per-frame emission probs.
    emission_probs: (T, N) matrix of P(observation_t | state_n).
    Returns: list of state indices (length T).
    """
    T, N = emission_probs.shape
    log_start = np.log(startprob + 1e-12)
    log_trans = np.log(transmat + 1e-12)
    log_emiss = np.log(emission_probs + 1e-12)

    # Viterbi tables
    V = np.zeros((T, N))
    ptr = np.zeros((T, N), dtype=int)

    V[0] = log_start + log_emiss[0]

    for t in range(1, T):
        for j in range(N):
            candidates = V[t - 1] + log_trans[:, j]
            ptr[t, j] = np.argmax(candidates)
            V[t, j] = candidates[ptr[t, j]] + log_emiss[t, j]

    # Backtrace
    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(V[-1])
    for t in range(T - 2, -1, -1):
        path[t] = ptr[t + 1, path[t + 1]]

    return path


# ── Guitar voicing system for tablature ──

# E-form barre shapes (offsets from root fret on 6th string)
_E_BARRE = {
    '':     [0, 2, 2, 1, 0, 0],
    'm':    [0, 2, 2, 0, 0, 0],
    '7':    [0, 2, 0, 1, 0, 0],
    'm7':   [0, 2, 0, 0, 0, 0],
    '7M':   [0, 2, 1, 1, 0, 0],
    'dim':  [0, 1, 2, 0, -1, -1],
    'aug':  [0, 3, 2, 1, 1, 0],
    'sus2': [0, 2, 4, 4, 0, 0],
    'sus4': [0, 2, 2, 2, 0, 0],
}

# A-form barre shapes (offsets from root fret on 5th string)
_A_BARRE = {
    '':     [-1, 0, 2, 2, 2, 0],
    'm':    [-1, 0, 2, 2, 1, 0],
    '7':    [-1, 0, 2, 0, 2, 0],
    'm7':   [-1, 0, 2, 0, 1, 0],
    '7M':   [-1, 0, 2, 1, 2, 0],
    'dim':  [-1, 0, 1, 2, 1, -1],
    'aug':  [-1, 0, 3, 2, 2, 1],
    'sus2': [-1, 0, 2, 2, 0, 0],
    'sus4': [-1, 0, 2, 2, 3, 0],
}

# Preferred open voicings (sound better than barre equivalents)
_OPEN_VOICINGS = {
    'C': [-1, 3, 2, 0, 1, 0], 'C7': [-1, 3, 2, 3, 1, 0],
    'D': [-1, -1, 0, 2, 3, 2], 'Dm': [-1, -1, 0, 2, 3, 1],
    'D7': [-1, -1, 0, 2, 1, 2], 'Dm7': [-1, -1, 0, 2, 1, 1],
    'D7M': [-1, -1, 0, 2, 2, 2],
    'Dsus2': [-1, -1, 0, 2, 3, 0], 'Dsus4': [-1, -1, 0, 2, 3, 3],
    'E': [0, 2, 2, 1, 0, 0], 'Em': [0, 2, 2, 0, 0, 0],
    'E7': [0, 2, 0, 1, 0, 0], 'Em7': [0, 2, 0, 0, 0, 0],
    'E7M': [0, 2, 1, 1, 0, 0], 'Esus4': [0, 2, 2, 2, 0, 0],
    'G': [3, 2, 0, 0, 0, 3], 'G7': [3, 2, 0, 0, 0, 1],
    'A': [-1, 0, 2, 2, 2, 0], 'Am': [-1, 0, 2, 2, 1, 0],
    'A7': [-1, 0, 2, 0, 2, 0], 'Am7': [-1, 0, 2, 0, 1, 0],
    'A7M': [-1, 0, 2, 1, 2, 0],
    'Asus2': [-1, 0, 2, 2, 0, 0], 'Asus4': [-1, 0, 2, 2, 3, 0],
}

_ROOT_FRET_6 = {n: (i - 4) % 12 for i, n in enumerate(NOTE_NAMES)}
_ROOT_FRET_5 = {n: (i - 9) % 12 for i, n in enumerate(NOTE_NAMES)}


def _get_voicing(chord_name: str) -> list[int]:
    """Get guitar voicing [E,A,D,G,B,e] for a chord. -1 = muted."""
    if chord_name in _OPEN_VOICINGS:
        return _OPEN_VOICINGS[chord_name]
    root = _extract_root(chord_name)
    suffix = chord_name[len(root):]
    fret_6 = _ROOT_FRET_6.get(root, 0)
    fret_5 = _ROOT_FRET_5.get(root, 0)
    if fret_5 < fret_6 and suffix in _A_BARRE:
        shape = _A_BARRE[suffix]
        base = fret_5
    else:
        shape = _E_BARRE.get(suffix, _E_BARRE[''])
        base = fret_6
    return [f + base if f >= 0 else -1 for f in shape]


def generate_tablature(chords: list[dict], key: str = '?', bpm: float = 0) -> str:
    """Generate text-based guitar tablature from chord data."""
    if not chords:
        return ''

    lines: list[str] = []
    lines.append('=' * 58)
    lines.append('           BeatScope  \u2014  Guitar Tablature')
    lines.append('=' * 58)
    lines.append('')
    lines.append(f'  Tom: {key}  |  BPM: {bpm}')
    lines.append('  Afinacao padrao: E A D G B e')
    lines.append('')

    # Unique chords
    unique: list[str] = []
    seen: set[str] = set()
    for c in chords:
        name = c['chord']
        if name not in seen:
            unique.append(name)
            seen.add(name)

    lines.append(f'  Acordes: {" - ".join(unique)}')
    lines.append('')
    lines.append('--- Diagramas ' + '-' * 43)
    lines.append('')

    for i in range(0, len(unique), 2):
        pair = unique[i:i + 2]
        parts = []
        for name in pair:
            v = _get_voicing(name)
            frets = ' '.join('x' if f == -1 else str(f) for f in v)
            parts.append(f'  {name:8s} {frets}')
        lines.append('    '.join(parts))
    lines.append(f'  {"":8s} E A D G B e')
    lines.append('')
    lines.append('--- Tablatura ' + '-' * 43)
    lines.append('')

    PER_ROW = 4
    COL = 17

    for row_start in range(0, len(chords), PER_ROW):
        row = chords[row_start:row_start + PER_ROW]
        header = '    '
        for c in row:
            header += c['chord'].ljust(COL)
        lines.append(header)

        string_labels = ['e', 'B', 'G', 'D', 'A', 'E']
        for si, sl in enumerate(string_labels):
            line = f'{sl}|'
            for c in row:
                v = _get_voicing(c['chord'])
                fret = v[5 - si]
                if fret == -1:
                    fs = '--x'
                elif fret >= 10:
                    fs = f'-{fret}'
                else:
                    fs = f'--{fret}'
                line += fs + '-' * (COL - len(fs) - 1) + '|'
            lines.append(line)

        tline = '    '
        for c in row:
            t = c['time']
            m = int(t) // 60
            s = int(t) % 60
            tline += f'{m}:{s:02d}'.ljust(COL)
        lines.append(tline)
        lines.append('')

    return '\n'.join(lines)


def _generate_tab_pdf(chords: list[dict], key: str, bpm) -> io.BytesIO:
    """Generate a professional guitar tablature PDF using reportlab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas
    from reportlab.lib.colors import HexColor

    buf = io.BytesIO()
    W, H = A4
    c = canvas.Canvas(buf, pagesize=A4)

    # Colors
    BG        = HexColor('#0a0a0f')
    SURFACE   = HexColor('#15151f')
    ACCENT    = HexColor('#7fff6e')
    ACCENT2   = HexColor('#ff6eb4')
    ACCENT3   = HexColor('#6eb4ff')
    TEXT      = HexColor('#e8e8f0')
    MUTED     = HexColor('#6a6a8a')
    BORDER    = HexColor('#2a2a3e')

    MONO = 'Courier'
    SANS = 'Helvetica'

    MARGIN_L = 30 * mm
    MARGIN_R = 15 * mm
    TAB_W = W - MARGIN_L - MARGIN_R

    PER_ROW = 4
    COL_W = TAB_W / PER_ROW

    # Unique chords for the diagram section
    unique: list[str] = []
    seen: set[str] = set()
    for ch in chords:
        name = ch['chord']
        if name not in seen:
            unique.append(name)
            seen.add(name)

    def new_page():
        c.setFillColor(BG)
        c.rect(0, 0, W, H, fill=1, stroke=0)

    def draw_header(y_pos):
        # Title bar
        c.setFillColor(SURFACE)
        c.roundRect(MARGIN_L - 10 * mm, y_pos - 8 * mm, TAB_W + 20 * mm, 22 * mm, 3 * mm, fill=1, stroke=0)

        c.setFont(SANS + '-Bold', 20)
        c.setFillColor(ACCENT)
        c.drawString(MARGIN_L, y_pos, 'BeatScope')
        c.setFillColor(TEXT)
        c.drawString(MARGIN_L + c.stringWidth('BeatScope', SANS + '-Bold', 20) + 4, y_pos, '— Guitar Tablature')

        y_pos -= 14 * mm
        c.setFont(MONO, 9)
        c.setFillColor(MUTED)
        c.drawString(MARGIN_L, y_pos, f'Tom: {key}   |   BPM: {bpm}   |   Afinação padrão: E A D G B e')

        y_pos -= 6 * mm
        c.setFillColor(MUTED)
        c.drawString(MARGIN_L, y_pos, f'Acordes: {" - ".join(unique)}')

        return y_pos - 8 * mm

    def draw_chord_diagrams(y_pos):
        c.setFont(SANS + '-Bold', 10)
        c.setFillColor(ACCENT2)
        c.drawString(MARGIN_L, y_pos, 'DIAGRAMAS DE ACORDES')
        y_pos -= 5 * mm
        c.setStrokeColor(BORDER)
        c.setLineWidth(0.5)
        c.line(MARGIN_L, y_pos, W - MARGIN_R, y_pos)
        y_pos -= 8 * mm

        cols = 4
        box_w = TAB_W / cols
        for idx, name in enumerate(unique):
            col = idx % cols
            if idx > 0 and col == 0:
                y_pos -= 18 * mm
                if y_pos < 30 * mm:
                    c.showPage()
                    new_page()
                    y_pos = H - 25 * mm

            x = MARGIN_L + col * box_w
            v = _get_voicing(name)
            frets = '  '.join('x' if f == -1 else str(f) for f in v)

            c.setFont(SANS + '-Bold', 10)
            c.setFillColor(ACCENT3)
            c.drawString(x, y_pos, name)

            c.setFont(MONO, 8.5)
            c.setFillColor(TEXT)
            c.drawString(x, y_pos - 5 * mm, frets)

            c.setFont(MONO, 6.5)
            c.setFillColor(MUTED)
            c.drawString(x, y_pos - 9 * mm, 'E  A  D  G  B  e')

        y_pos -= 20 * mm
        return y_pos

    def draw_tab_section(y_pos, row_chords):
        """Draw one row of tablature (up to PER_ROW chords)."""
        needed = 24 * mm
        if y_pos - needed < 20 * mm:
            c.showPage()
            new_page()
            y_pos = H - 25 * mm

        # Chord names header
        c.setFont(SANS + '-Bold', 9)
        for i, ch in enumerate(row_chords):
            x = MARGIN_L + i * COL_W
            c.setFillColor(ACCENT2)
            c.drawString(x + 4, y_pos, ch['chord'])

        y_pos -= 5 * mm

        # Tab lines
        string_labels = ['e', 'B', 'G', 'D', 'A', 'E']
        line_spacing = 2.6 * mm

        for si, sl in enumerate(string_labels):
            line_y = y_pos - si * line_spacing

            # String label
            c.setFont(MONO, 7)
            c.setFillColor(MUTED)
            c.drawString(MARGIN_L - 7 * mm, line_y - 1, sl)

            # Staff line
            c.setStrokeColor(BORDER)
            c.setLineWidth(0.4)
            c.line(MARGIN_L, line_y, MARGIN_L + len(row_chords) * COL_W, line_y)

            # Fret numbers
            c.setFont(MONO, 8)
            c.setFillColor(TEXT)
            for i, ch in enumerate(row_chords):
                v = _get_voicing(ch['chord'])
                fret = v[5 - si]
                txt = 'x' if fret == -1 else str(fret)
                x = MARGIN_L + i * COL_W + COL_W * 0.35
                # Background circle for readability
                tw = c.stringWidth(txt, MONO, 8)
                c.setFillColor(BG)
                c.rect(x - 1, line_y - 2, tw + 2, 5, fill=1, stroke=0)
                c.setFillColor(ACCENT if fret >= 0 else MUTED)
                c.drawString(x, line_y - 1.5, txt)

        y_pos -= len(string_labels) * line_spacing + 2 * mm

        # Timestamps
        c.setFont(MONO, 6.5)
        c.setFillColor(MUTED)
        for i, ch in enumerate(row_chords):
            t = ch['time']
            m = int(t) // 60
            s = int(t) % 60
            x = MARGIN_L + i * COL_W + 4
            c.drawString(x, y_pos, f'{m}:{s:02d}')

        y_pos -= 7 * mm
        return y_pos

    # === Build the PDF ===
    new_page()
    y = H - 25 * mm
    y = draw_header(y)
    y = draw_chord_diagrams(y)

    # Separator
    c.setFont(SANS + '-Bold', 10)
    c.setFillColor(ACCENT)
    c.drawString(MARGIN_L, y, 'TABLATURA')
    y -= 5 * mm
    c.setStrokeColor(BORDER)
    c.setLineWidth(0.5)
    c.line(MARGIN_L, y, W - MARGIN_R, y)
    y -= 8 * mm

    for row_start in range(0, len(chords), PER_ROW):
        row = chords[row_start:row_start + PER_ROW]
        y = draw_tab_section(y, row)

    # Footer on last page
    c.setFont(MONO, 7)
    c.setFillColor(MUTED)
    c.drawString(MARGIN_L, 10 * mm, 'Gerado por BeatScope')

    c.save()
    buf.seek(0)
    return buf


def extract_video_id(url: str) -> str | None:
    patterns = [
        r'[?&]v=([0-9A-Za-z_-]{11})',
        r'youtu\.be\/([0-9A-Za-z_-]{11})',
        r'youtube\.com\/shorts\/([0-9A-Za-z_-]{11})',
        r'youtube\.com\/embed\/([0-9A-Za-z_-]{11})',
        r'music\.youtube\.com\/watch\?v=([0-9A-Za-z_-]{11})',
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def detect_key(y, sr) -> tuple[str, float]:
    # 1. Separar harmônicos da percussão
    y_harmonic = librosa.effects.harmonic(y, margin=4)

    # 2. Usar múltiplos cromagramas e combinar
    chroma_cqt = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, bins_per_octave=36)
    chroma_stft = librosa.feature.chroma_stft(y=y_harmonic, sr=sr)
    chroma_cens = librosa.feature.chroma_cens(y=y_harmonic, sr=sr)

    # Média ponderada dos 3 cromagramas
    chroma_mean = (
        chroma_cqt.mean(axis=1) * 0.5 +
        chroma_stft.mean(axis=1) * 0.3 +
        chroma_cens.mean(axis=1) * 0.2
    )

    # 3. Normalizar antes de correlacionar
    chroma_mean = chroma_mean / (np.linalg.norm(chroma_mean) + 1e-6)

    major_scores = []
    minor_scores = []
    for i in range(12):
        rotated = np.roll(chroma_mean, -i)
        major_scores.append(np.corrcoef(rotated, MAJOR_PROFILE)[0, 1])
        minor_scores.append(np.corrcoef(rotated, MINOR_PROFILE)[0, 1])

    best_major = max(range(12), key=lambda i: major_scores[i])
    best_minor = max(range(12), key=lambda i: minor_scores[i])

    if major_scores[best_major] >= minor_scores[best_minor]:
        return f"{NOTE_NAMES[best_major]} Major", float(major_scores[best_major])
    else:
        return f"{NOTE_NAMES[best_minor]} Minor", float(minor_scores[best_minor])


def detect_bpm(y, sr) -> tuple[float, list[float]]:
    # 1. Separar percussão para melhor detecção de beats
    y_percussive = librosa.effects.percussive(y, margin=4)

    onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr)

    # 2. Múltiplos estimadores
    tempo_default = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]

    pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
    tempo_plp = librosa.beat.tempo(onset_envelope=pulse, sr=sr)[0]

    # 3. Usar tempograma para estimativa mais robusta
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
    tempo_tg = librosa.beat.tempo(onset_envelope=onset_env, sr=sr,
                                   prior=None, aggregate=np.median)[0]

    # 4. Mediana ao invés de média (mais robusta a outliers)
    candidates = [tempo_default, tempo_plp, tempo_tg]
    final_tempo = round(float(np.median(candidates)), 1)

    # 5. Corrigir subharmônicos: se BPM < 60, provavelmente é o dobro
    if final_tempo < 60:
        final_tempo *= 2
    elif final_tempo > 200:
        final_tempo /= 2

    _, beats = librosa.beat.beat_track(y=y_percussive, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr).tolist()

    return round(final_tempo, 1), beat_times[:20]

def analyze_audio(audio_path: str) -> dict:
    """Run full audio analysis on a file."""
    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    bpm, beats = detect_bpm(y, sr)
    key, confidence = detect_key(y, sr)
    chords = detect_chords(y, sr)

    return {
        "bpm": bpm,
        "key": key,
        "key_confidence": round(confidence * 100, 1),
        "beats": beats,
        "chords": chords,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/audio/<path:filename>")
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename)


@app.route("/generate_tab", methods=["POST"])
def gen_tab():
    data = request.get_json()
    chords = data.get("chords", [])
    key = data.get("key", "?")
    bpm = data.get("bpm", "?")

    if not chords:
        return jsonify({"error": "Nenhum acorde detectado."}), 400

    buf = _generate_tab_pdf(chords, key, bpm)
    return send_file(buf, mimetype='application/pdf',
                     as_attachment=True, download_name='tablature.pdf')


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    url = data.get("url", "").strip()

    if not url:
        return jsonify({"error": "URL não fornecida."}), 400

    video_id = extract_video_id(url)
    if not video_id:
        return jsonify({"error": "URL do YouTube inválida."}), 400

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.%(ext)s")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": audio_path,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }],
            "quiet": True,
            "no_warnings": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        except Exception as e:
            return jsonify({"error": f"Erro ao baixar áudio: {str(e)}"}), 500

        wav_path = os.path.join(tmpdir, "audio.wav")
        if not os.path.exists(wav_path):
            files = os.listdir(tmpdir)
            if not files:
                return jsonify({"error": "Falha ao extrair áudio."}), 500
            wav_path = os.path.join(tmpdir, files[0])

        try:
            result = analyze_audio(wav_path)
            # Copy audio to cache so the frontend can play it
            cached_name = f"{video_id}.wav"
            shutil.copy2(wav_path, os.path.join(AUDIO_DIR, cached_name))
            result["audio_url"] = f"/audio/{cached_name}"
            result["video_id"] = video_id
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": f"Erro na análise: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)