from flask import Flask, render_template_string, request, jsonify
import math
import time
from collections import deque
import heapq

app = Flask(__name__)

# ==============================================================================
# BAGIAN 1: LOGIKA AI SEARCH - WATER JUG PROBLEM (BFS, UCS, A*)
# ==============================================================================
# Mendukung 3 Operator Lengkap:
# 1. Fill  : Isi penuh teko i dari keran
# 2. Empty : Buang seluruh isi teko i (kosongkan)
# 3. Pour  : Pindahkan isi teko i ke teko j (hingga teko i kosong atau teko j penuh)
# Start State: Semua teko kosong (0, 0, ..., 0)
# Goal State : Target volume air tercapai di salah satu teko (single goal)
# ==============================================================================

def get_neighbors(state, caps):
    """
    Membangkitkan semua transisi aksi yang valid dari state saat ini:
    1. Isi penuh teko i dari keran (Fill)
    2. Tuang dari teko i ke teko j (Pour)
    3. Buang seluruh isi teko i (Empty)
    Returns: list of (nxt_state, amount, action_text, src_idx, dst_idx, action_type)
    """
    n = len(state)
    neighbors = []

    # 1. AKSI ISI PENUH DARI KERAN (FILL)
    for i in range(n):
        if state[i] < caps[i]:
            added = caps[i] - state[i]
            nxt = list(state)
            nxt[i] = caps[i]
            action_text = f"Isi penuh Teko {i+1} ({caps[i]}L) dari keran"
            neighbors.append((tuple(nxt), added, action_text, None, i, "fill"))

    # 2. AKSI PINDAH/TUANG ANTAR TEKO (POUR)
    for i in range(n):
        if state[i] == 0:
            continue
        for j in range(n):
            if i == j:
                continue
            if state[j] < caps[j]:
                amount = min(state[i], caps[j] - state[j])
                nxt = list(state)
                nxt[i] -= amount
                nxt[j] += amount
                
                rem_src = nxt[i]
                curr_dst = nxt[j]
                cap_dst = caps[j]
                status_dst = "hingga penuh" if curr_dst == cap_dst else ""
                status_src = "hingga kosong" if rem_src == 0 else f"(sisa {rem_src}L)"
                
                action_text = f"Tuang {amount}L dari Teko {i+1} ke Teko {j+1} {status_dst} {status_src}".strip()
                neighbors.append((tuple(nxt), amount, action_text, i, j, "pour"))

    # 3. AKSI BUANG SELURUH AIR (EMPTY)
    for i in range(n):
        if state[i] > 0:
            emptied = state[i]
            nxt = list(state)
            nxt[i] = 0
            action_text = f"Buang seluruh isi Teko {i+1} ({emptied}L dibuang)"
            neighbors.append((tuple(nxt), emptied, action_text, i, None, "empty"))

    return neighbors

def make_goal_checker(target_amount):
    """Goal tercapai bila salah satu teko memiliki volume tepat target_amount."""
    return lambda s: any(v == target_amount for v in s)

def make_heuristic(target_amount):
    """
    Heuristik Admisibel A*:
    h(n) = min(|v - target_amount| for v in s)
    Karena perpindahan air per aksi maksimal mengubah volume teko sebesar selisihnya,
    nilai ini adalah lower bound liter menuju target terdekat.
    """
    return lambda s: float(min(abs(v - target_amount) for v in s))

def validate_inputs(caps, target_amount):
    n = len(caps)
    if not (2 <= n <= 5):
        return False, f"Jumlah teko harus antara 2 sampai 5 teko (input: {n} teko)!"
    
    for i in range(n):
        if caps[i] <= 0:
            return False, f"Kapasitas Teko {i+1} harus lebih besar dari 0 (> 0)!"

    if target_amount <= 0:
        return False, "Target volume air yang ingin diukur harus lebih besar dari 0 L!"
    
    max_c = max(caps)
    if target_amount > max_c:
        return False, f"Target ({target_amount}L) tidak boleh melebihi kapasitas teko terbesar ({max_c}L)!"

    # FPB (GCD) feasibility check
    overall_gcd = caps[0]
    for c in caps[1:]:
        overall_gcd = math.gcd(overall_gcd, c)
    if target_amount % overall_gcd != 0:
        return False, f"Target ({target_amount}L) tidak mungkin dicapai karena bukan kelipatan FPB kapasitas teko (FPB = {overall_gcd})!"

    return True, "Valid"

# 1. BREADTH-FIRST SEARCH (BFS)
def solve_bfs(caps, start, is_goal, h_func):
    t0 = time.perf_counter()
    start_t = tuple(start)
    
    if is_goal(start_t):
        return {
            "name": "BFS (Breadth-First Search)",
            "found": True,
            "path": [{"state": list(start_t), "action": "Kondisi Awal (Target Sudah Tercapai)", "amount": 0, "g": 0, "h": 0, "f": 0, "src": None, "dst": None, "action_type": "init"}],
            "steps": 0,
            "total_liters": 0,
            "expanded": 0,
            "generated": 1,
            "max_open": 1,
            "time_ms": 0.001,
            "open_sample": [list(start_t)],
            "closed_sample": []
        }

    queue = deque([start_t])
    visited = {start_t}
    came_from = {start_t: (None, "Mulai (Semua Teko Kosong)", 0, None, None, "start")}
    g_score = {start_t: 0}
    
    expanded = 0
    generated = 1
    max_open = 1
    found = False
    goal_found = None
    
    while queue and expanded < 100000:
        max_open = max(max_open, len(queue))
        curr = queue.popleft()
        expanded += 1
        
        if is_goal(curr):
            found = True
            goal_found = curr
            break
            
        for nxt, amount, action, src, dst, action_type in get_neighbors(curr, caps):
            if nxt not in visited:
                visited.add(nxt)
                came_from[nxt] = (curr, action, amount, src, dst, action_type)
                g_score[nxt] = g_score[curr] + amount
                queue.append(nxt)
                generated += 1
                
    time_ms = round((time.perf_counter() - t0) * 1000, 3)
    
    if not found:
        return {
            "name": "BFS (Breadth-First Search)",
            "found": False,
            "path": [],
            "steps": 0,
            "total_liters": 0,
            "expanded": expanded,
            "generated": generated,
            "max_open": max_open,
            "time_ms": time_ms,
            "open_sample": [list(s) for s in list(queue)[:10]],
            "closed_sample": [list(s) for s in list(visited)[:10]]
        }
        
    path = []
    curr = goal_found
    while curr is not None:
        prev, action, amount, src, dst, action_type = came_from[curr]
        h_val = round(h_func(curr), 1)
        g_val = g_score[curr]
        path.append({
            "state": list(curr),
            "action": action,
            "amount": amount,
            "g": g_val,
            "h": h_val,
            "f": round(g_val + h_val, 1),
            "src": src,
            "dst": dst,
            "action_type": action_type
        })
        curr = prev
    path.reverse()
    
    return {
        "name": "BFS (Breadth-First Search)",
        "found": True,
        "path": path,
        "steps": max(0, len(path) - 1),
        "total_liters": g_score[goal_found],
        "expanded": expanded,
        "generated": generated,
        "max_open": max_open,
        "time_ms": time_ms,
        "open_sample": [list(s) for s in list(queue)[:10]],
        "closed_sample": [list(s) for s in list(visited)[:10]]
    }

# 2. UNIFORM-COST SEARCH (UCS)
def solve_ucs(caps, start, is_goal, h_func):
    t0 = time.perf_counter()
    start_t = tuple(start)
    
    if is_goal(start_t):
        return {
            "name": "UCS (Uniform-Cost Search)",
            "found": True,
            "path": [{"state": list(start_t), "action": "Kondisi Awal (Target Sudah Tercapai)", "amount": 0, "g": 0, "h": 0, "f": 0, "src": None, "dst": None, "action_type": "init"}],
            "steps": 0,
            "total_liters": 0,
            "expanded": 0,
            "generated": 1,
            "max_open": 1,
            "time_ms": 0.001,
            "open_sample": [list(start_t)],
            "closed_sample": []
        }

    counter = 0
    pq = [(0, counter, start_t)]
    came_from = {start_t: (None, "Mulai (Semua Teko Kosong)", 0, None, None, "start")}
    g_score = {start_t: 0}
    closed = set()
    
    expanded = 0
    generated = 1
    max_open = 1
    found = False
    goal_found = None
    
    while pq and expanded < 100000:
        max_open = max(max_open, len(pq))
        curr_g, _, curr = heapq.heappop(pq)
        
        if curr in closed:
            continue
        closed.add(curr)
        expanded += 1
        
        if is_goal(curr):
            found = True
            goal_found = curr
            break
            
        for nxt, amount, action, src, dst, action_type in get_neighbors(curr, caps):
            tentative_g = curr_g + amount
            if nxt not in g_score or tentative_g < g_score[nxt]:
                g_score[nxt] = tentative_g
                came_from[nxt] = (curr, action, amount, src, dst, action_type)
                counter += 1
                heapq.heappush(pq, (tentative_g, counter, nxt))
                generated += 1
                
    time_ms = round((time.perf_counter() - t0) * 1000, 3)
    
    if not found:
        return {
            "name": "UCS (Uniform-Cost Search)",
            "found": False,
            "path": [],
            "steps": 0,
            "total_liters": 0,
            "expanded": expanded,
            "generated": generated,
            "max_open": max_open,
            "time_ms": time_ms,
            "open_sample": [list(item[2]) for item in pq[:10]],
            "closed_sample": [list(s) for s in list(closed)[:10]]
        }
        
    path = []
    curr = goal_found
    while curr is not None:
        prev, action, amount, src, dst, action_type = came_from[curr]
        h_val = round(h_func(curr), 1)
        g_val = g_score[curr]
        path.append({
            "state": list(curr),
            "action": action,
            "amount": amount,
            "g": g_val,
            "h": h_val,
            "f": round(g_val + h_val, 1),
            "src": src,
            "dst": dst,
            "action_type": action_type
        })
        curr = prev
    path.reverse()
    
    return {
        "name": "UCS (Uniform-Cost Search)",
        "found": True,
        "path": path,
        "steps": max(0, len(path) - 1),
        "total_liters": g_score[goal_found],
        "expanded": expanded,
        "generated": generated,
        "max_open": max_open,
        "time_ms": time_ms,
        "open_sample": [list(item[2]) for item in pq[:10]],
        "closed_sample": [list(s) for s in list(closed)[:10]]
    }

# 3. A* SEARCH (A-STAR)
def solve_astar(caps, start, is_goal, h_func):
    t0 = time.perf_counter()
    start_t = tuple(start)
    
    if is_goal(start_t):
        return {
            "name": "A* Search",
            "found": True,
            "path": [{"state": list(start_t), "action": "Kondisi Awal (Target Sudah Tercapai)", "amount": 0, "g": 0, "h": 0, "f": 0, "src": None, "dst": None, "action_type": "init"}],
            "steps": 0,
            "total_liters": 0,
            "expanded": 0,
            "generated": 1,
            "max_open": 1,
            "time_ms": 0.001,
            "open_sample": [list(start_t)],
            "closed_sample": []
        }

    counter = 0
    h_start = h_func(start_t)
    pq = [(h_start, counter, start_t)]
    came_from = {start_t: (None, "Mulai (Semua Teko Kosong)", 0, None, None, "start")}
    g_score = {start_t: 0}
    closed = set()
    
    expanded = 0
    generated = 1
    max_open = 1
    found = False
    goal_found = None
    
    while pq and expanded < 100000:
        max_open = max(max_open, len(pq))
        curr_f, _, curr = heapq.heappop(pq)
        
        if curr in closed:
            continue
        closed.add(curr)
        expanded += 1
        
        if is_goal(curr):
            found = True
            goal_found = curr
            break
            
        for nxt, amount, action, src, dst, action_type in get_neighbors(curr, caps):
            tentative_g = g_score[curr] + amount
            if nxt not in g_score or tentative_g < g_score[nxt]:
                g_score[nxt] = tentative_g
                h_nxt = h_func(nxt)
                f_nxt = tentative_g + h_nxt
                came_from[nxt] = (curr, action, amount, src, dst, action_type)
                counter += 1
                heapq.heappush(pq, (f_nxt, counter, nxt))
                generated += 1
                
    time_ms = round((time.perf_counter() - t0) * 1000, 3)
    
    if not found:
        return {
            "name": "A* Search",
            "found": False,
            "path": [],
            "steps": 0,
            "total_liters": 0,
            "expanded": expanded,
            "generated": generated,
            "max_open": max_open,
            "time_ms": time_ms,
            "open_sample": [list(item[2]) for item in pq[:10]],
            "closed_sample": [list(s) for s in list(closed)[:10]]
        }
        
    path = []
    curr = goal_found
    while curr is not None:
        prev, action, amount, src, dst, action_type = came_from[curr]
        h_val = round(h_func(curr), 1)
        g_val = g_score[curr]
        path.append({
            "state": list(curr),
            "action": action,
            "amount": amount,
            "g": g_val,
            "h": h_val,
            "f": round(g_val + h_val, 1),
            "src": src,
            "dst": dst,
            "action_type": action_type
        })
        curr = prev
    path.reverse()
    
    return {
        "name": "A* Search",
        "found": True,
        "path": path,
        "steps": max(0, len(path) - 1),
        "total_liters": g_score[goal_found],
        "expanded": expanded,
        "generated": generated,
        "max_open": max_open,
        "time_ms": time_ms,
        "open_sample": [list(item[2]) for item in pq[:10]],
        "closed_sample": [list(s) for s in list(closed)[:10]]
    }


# ==============================================================================
# BAGIAN 2: TEMPLATE HTML + INTERACTIVE JUG PUZZLE & AI SEARCH VISUALIZER
# ==============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Water Jug Puzzle & AI Search Visualizer (BFS, UCS, A*)</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-page: #090e17;
            --card-bg: #111827;
            --stage-bg: #0d1527;
            --border-color: #1e293b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-cyan: #38bdf8;
            --accent-blue: #0284c7;
            --accent-green: #10b981;
            --accent-amber: #f59e0b;
            --accent-rose: #f43f5e;
            --accent-purple: #a855f7;
            --water-color: #0284c7;
            --water-glow: rgba(56, 189, 248, 0.4);
        }

        * { box-sizing: border-box; font-family: 'Outfit', sans-serif; }
        code, pre, .mono { font-family: 'JetBrains Mono', monospace; }

        body {
            margin: 0; padding: 24px 16px;
            background-color: var(--bg-page);
            color: var(--text-main);
            min-height: 100vh;
            display: flex; flex-direction: column; align-items: center;
        }

        .container {
            max-width: 1200px;
            width: 100%;
            display: flex;
            flex-direction: column;
            gap: 18px;
        }

        /* Header */
        header {
            text-align: center;
            padding: 4px 0 8px 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 6px;
        }
        .header-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background-color: #1e293b;
            color: var(--accent-cyan);
            border: 1px solid #334155;
            border-radius: 9999px;
            padding: 4px 14px;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.5px;
        }
        h1 {
            margin: 0;
            font-size: 28px;
            font-weight: 800;
            color: var(--text-main);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
        }
        p.subtitle {
            margin: 0;
            color: var(--text-muted);
            font-size: 14px;
            max-width: 720px;
            line-height: 1.5;
        }

        /* Main Mode Switcher */
        .mode-switcher-container {
            display: flex;
            justify-content: center;
            margin: 4px 0;
        }
        /* AI Mode Header */
        .ai-mode-header {
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 0 16px 0;
        }
        .ai-mode-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 20px;
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(2, 132, 199, 0.18), rgba(56, 189, 248, 0.08));
            border: 1px solid rgba(56, 189, 248, 0.35);
            color: var(--accent-cyan);
            font-size: 13px;
            font-weight: 700;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.25);
            letter-spacing: 0.2px;
        }

        /* Solid Cards */
        .solid-card {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 20px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        }

        /* Preset Chips */
        .presets-bar {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 0px;
            padding-bottom: 0px;
            border-bottom: none;
            transition: all 0.2s ease;
        }
        .presets-bar.with-border {
            margin-bottom: 16px;
            padding-bottom: 14px;
            border-bottom: 1px solid #1e293b;
        }
        .preset-chip {
            background-color: #1e293b;
            border: 1px solid #334155;
            color: var(--text-main);
            padding: 7px 13px;
            border-radius: 8px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            transition: all 0.15s ease;
            user-select: none;
        }
        .preset-chip:hover {
            border-color: var(--accent-cyan);
            background-color: #273549;
            transform: translateY(-1px);
        }
        .preset-chip.highlight {
            border-color: var(--accent-amber);
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.22), rgba(245, 158, 11, 0.1));
            color: #fbbf24;
            box-shadow: 0 0 12px rgba(245, 158, 11, 0.25);
            font-weight: 700;
        }
        .chip-badge {
            background-color: rgba(15, 23, 42, 0.85);
            border: 1px solid rgba(56, 189, 248, 0.3);
            color: var(--accent-cyan);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: 800;
        }
        .preset-info-card {
            margin-top: 12px;
            padding: 12px 16px;
            background-color: #0b1120;
            border: 1px solid #1e293b;
            border-left: 3px solid var(--accent-amber);
            border-radius: 8px;
            display: flex;
            flex-direction: column;
            gap: 5px;
            transition: all 0.2s ease;
        }
        .preset-info-header {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .preset-info-tag {
            font-size: 10px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            padding: 2px 8px;
            border-radius: 4px;
            background: rgba(245, 158, 11, 0.15);
            color: #fbbf24;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }
        .preset-info-title {
            font-size: 13px;
            font-weight: 700;
            color: var(--text-main);
        }
        .preset-info-desc {
            font-size: 12px;
            color: var(--text-muted);
            margin: 0;
            line-height: 1.45;
        }

        /* Target Mode Selector Tabs */
        .target-mode-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 12px;
            margin-bottom: 16px;
        }
        .target-pills {
            display: flex;
            background-color: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 3px;
            gap: 4px;
        }
        .target-pill-btn {
            background-color: transparent;
            border: none;
            color: var(--text-muted);
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .target-pill-btn.active {
            background-color: #1e293b;
            color: var(--accent-cyan);
        }

        /* Dual Start & Target Controls Grid */
        .target-and-start-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-bottom: 16px;
        }
        @media (max-width: 768px) {
            .target-and-start-grid {
                grid-template-columns: 1fr;
            }
        }
        .single-target-box {
            background-color: rgba(56, 189, 248, 0.06);
            border: 1px dashed rgba(56, 189, 248, 0.35);
            border-radius: 10px;
            padding: 12px 16px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .single-target-box.start-box {
            background-color: rgba(245, 158, 11, 0.06);
            border: 1px dashed rgba(245, 158, 11, 0.35);
        }
        .single-target-title {
            font-size: 13px;
            font-weight: 700;
            color: var(--text-main);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .single-target-control {
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }
        .input-target-amt {
            width: 70px;
            font-size: 16px;
            font-weight: 800;
            text-align: center;
            background-color: #0f172a;
            border: 1px solid var(--accent-cyan);
            border-radius: 8px;
            padding: 6px 8px;
            color: var(--accent-cyan);
            font-family: 'JetBrains Mono', monospace;
            outline: none;
        }
        .input-target-amt.start-input {
            border-color: var(--accent-amber);
            color: #fbbf24;
        }

        /* Jugs Inputs Grid */
        .jugs-input-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
            margin-bottom: 14px;
        }
        .jug-input-card {
            background-color: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 10px;
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .jug-input-header {
            font-size: 13px;
            font-weight: 700;
            color: var(--accent-cyan);
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 6px;
        }
        .teko-main-tag {
            font-size: 10px;
            font-weight: 700;
            color: var(--accent-cyan);
            background: rgba(56, 189, 248, 0.12);
            border: 1px solid rgba(56, 189, 248, 0.3);
            padding: 1px 6px;
            border-radius: 4px;
        }
        .auto-fill-info {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 12px;
            color: var(--text-muted);
            font-weight: 600;
            margin-top: 2px;
        }
        .badge-auto-fill {
            font-size: 11px;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 6px;
            display: inline-flex;
            align-items: center;
        }
        .badge-auto-fill.filled {
            background-color: rgba(56, 189, 248, 0.15);
            color: var(--accent-cyan);
            border: 1px solid rgba(56, 189, 248, 0.3);
        }
        .badge-auto-fill.empty {
            background-color: #1e293b;
            color: var(--text-muted);
            border: 1px solid #334155;
        }
        .row-input {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            font-size: 12px;
            color: var(--text-muted);
            font-weight: 600;
        }
        .row-input input {
            width: 75px;
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 6px;
            padding: 5px 8px;
            color: var(--text-main);
            font-size: 13px;
            font-weight: 700;
            outline: none;
            font-family: 'JetBrains Mono', monospace;
            text-align: right;
        }
        .row-input input:focus {
            border-color: var(--accent-cyan);
        }

        /* Water Conservation Bar */
        .water-sum-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 10px;
            background-color: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 8px 14px;
            font-size: 13px;
            font-weight: 600;
        }
        .status-valid {
            color: var(--accent-green);
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .status-invalid {
            color: var(--accent-rose);
            display: flex;
            align-items: center;
            gap: 6px;
        }

        /* Error Notification Banner */
        #error-banner {
            display: none;
            background-color: rgba(244, 63, 94, 0.12);
            border: 1px solid var(--accent-rose);
            color: #fda4af;
            padding: 10px 14px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            align-items: center;
            gap: 8px;
            margin-top: 10px;
        }

        /* Main Simulation / Gameplay Stage */
        .stage-card {
            background-color: var(--stage-bg);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        /* Stage Instructions / Status Topbar */
        .stage-hud {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 10px;
            background-color: #111827;
            border: 1px solid #1e293b;
            border-radius: 10px;
            padding: 10px 16px;
        }
        .hud-status {
            font-size: 14px;
            font-weight: 700;
            color: var(--text-main);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .hud-stats {
            display: flex;
            align-items: center;
            gap: 14px;
            font-size: 13px;
            font-weight: 600;
            color: var(--text-muted);
        }
        .stat-badge {
            background-color: #1e293b;
            border: 1px solid #334155;
            padding: 3px 8px;
            border-radius: 6px;
            color: var(--accent-cyan);
            font-family: 'JetBrains Mono', monospace;
        }

        /* Visual Pouring Indicator Stream */
        .pouring-stream-indicator {
            height: 28px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            font-size: 13px;
            font-weight: 700;
            color: var(--accent-amber);
            opacity: 0;
            transition: opacity 0.2s ease;
        }
        .pouring-stream-indicator.active {
            opacity: 1;
        }

        /* Jugs Row Display */
        .jugs-row {
            display: flex;
            align-items: flex-end;
            justify-content: center;
            gap: 32px;
            width: 100%;
            flex-wrap: wrap;
            padding: 16px 0;
            min-height: 260px;
        }

        .jug-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
            position: relative;
            cursor: pointer;
            user-select: none;
            transition: transform 0.2s ease;
        }
        .jug-item:hover {
            transform: translateY(-4px);
        }
        .jug-item.selected-source {
            transform: translateY(-8px);
        }

        .jug-header-label {
            font-size: 14px;
            font-weight: 700;
            color: var(--text-main);
            display: flex;
            align-items: center;
            gap: 6px;
        }

        /* Glass Body */
        .jug-glass-body {
            position: relative;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.03) 0%, rgba(255, 255, 255, 0.08) 100%);
            border: 3px solid #64748b;
            border-top: none;
            border-radius: 0 0 16px 16px;
            overflow: hidden;
            display: flex;
            flex-direction: column-reverse;
            box-shadow: inset 0 0 12px rgba(0, 0, 0, 0.6);
            transition: all 0.25s ease;
        }

        .jug-item.selected-source .jug-glass-body {
            border-color: var(--accent-amber);
            box-shadow: 0 0 16px rgba(245, 158, 11, 0.4), inset 0 0 12px rgba(0, 0, 0, 0.6);
        }
        .jug-item.potential-target .jug-glass-body {
            border-color: var(--accent-green);
            box-shadow: 0 0 14px rgba(16, 185, 129, 0.35), inset 0 0 12px rgba(0, 0, 0, 0.6);
        }
        .jug-item.ai-source .jug-glass-body {
            border-color: var(--accent-amber) !important;
            box-shadow: 0 0 16px rgba(245, 158, 11, 0.5);
        }
        .jug-item.ai-target .jug-glass-body {
            border-color: var(--accent-green) !important;
            box-shadow: 0 0 16px rgba(16, 185, 129, 0.5);
        }
        .jug-item.ai-fill .jug-glass-body {
            border-color: var(--accent-cyan) !important;
            box-shadow: 0 0 18px rgba(56, 189, 248, 0.6);
        }
        .jug-item.ai-empty .jug-glass-body {
            border-color: var(--accent-rose) !important;
            box-shadow: 0 0 18px rgba(244, 63, 94, 0.6);
        }
        .jug-item.target-reached .jug-glass-body {
            border-color: #38bdf8 !important;
            box-shadow: 0 0 20px rgba(56, 189, 248, 0.7);
        }

        /* Water Level Column */
        .water-fill {
            width: 100%;
            height: 0%;
            background: linear-gradient(180deg, #38bdf8 0%, #0284c7 100%);
            transition: height 0.45s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
        }
        .water-wave-top {
            position: absolute;
            top: 0; left: 0; width: 100%; height: 4px;
            background-color: #7dd3fc;
            box-shadow: 0 0 8px #38bdf8;
        }

        /* Measuring Scale */
        .jug-measuring-scale {
            position: absolute;
            left: 0; top: 0; bottom: 0; width: 100%;
            pointer-events: none;
        }
        .scale-tick {
            position: absolute;
            left: 0;
            background-color: rgba(255, 255, 255, 0.35);
        }
        .scale-tick-major {
            width: 14px;
            height: 1.5px;
            background-color: rgba(255, 255, 255, 0.75);
        }
        .scale-tick-minor {
            width: 7px;
            height: 1px;
            background-color: rgba(255, 255, 255, 0.2);
        }
        .scale-label {
            position: absolute;
            left: 17px;
            font-size: 10px;
            font-weight: 700;
            color: rgba(255, 255, 255, 0.75);
            font-family: 'JetBrains Mono', monospace;
        }

        /* Jug Action Tag Badge */
        .jug-tag-badge {
            position: absolute;
            top: -24px;
            background-color: var(--accent-amber);
            color: #111827;
            font-size: 11px;
            font-weight: 800;
            padding: 2px 8px;
            border-radius: 9999px;
            display: none;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.4);
            white-space: nowrap;
        }
        .jug-item.selected-source .jug-tag-badge {
            display: block;
        }
        .jug-item.target-reached .jug-win-badge {
            display: block;
        }
        .jug-win-badge {
            position: absolute;
            top: -28px;
            background-color: var(--accent-green);
            color: white;
            font-size: 11px;
            font-weight: 800;
            padding: 3px 10px;
            border-radius: 9999px;
            display: none;
            box-shadow: 0 0 10px rgba(16, 185, 129, 0.6);
            white-space: nowrap;
        }

        .jug-badge-info {
            background-color: #111827;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 4px 10px;
            font-size: 13px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-main);
            text-align: center;
        }

        /* Interactive Controls Bar */
        .interactive-controls-bar {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            flex-wrap: wrap;
            padding: 8px 0;
            border-top: 1px solid #1e293b;
        }
        .btn-action {
            background-color: #1e293b;
            border: 1px solid #334155;
            color: var(--text-main);
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            transition: all 0.15s ease;
        }
        .btn-action:hover {
            background-color: #334155;
            border-color: var(--accent-cyan);
        }
        .btn-action:disabled {
            opacity: 0.4;
            cursor: not-allowed;
        }
        .btn-action.primary {
            background-color: var(--accent-blue);
            border-color: var(--accent-cyan);
            color: white;
        }
        .btn-action.primary:hover {
            background-color: #0369a1;
        }

        /* AI Narration & Step Info Card */
        .narration-card {
            background-color: #111827;
            border: 1px solid #1e293b;
            border-radius: 10px;
            padding: 12px 18px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .narration-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
        }
        .badge-step-pill {
            background-color: var(--accent-blue);
            color: white;
            font-size: 11px;
            font-weight: 800;
            padding: 2px 8px;
            border-radius: 4px;
            letter-spacing: 0.5px;
        }
        .narration-text {
            font-size: 14px;
            font-weight: 700;
            color: var(--text-main);
        }
        .node-eval-row {
            display: flex;
            gap: 14px;
            align-items: center;
            font-size: 12px;
            color: var(--text-muted);
            border-top: 1px solid #1e293b;
            padding-top: 6px;
            font-family: 'JetBrains Mono', monospace;
            flex-wrap: wrap;
        }
        .eval-val {
            color: var(--accent-cyan);
            font-weight: 700;
        }

        /* AI Playback Controls Toolbar */
        .ai-playback-toolbar {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            flex-wrap: wrap;
            width: 100%;
        }
        .timeline-slider-row {
            display: flex;
            align-items: center;
            gap: 10px;
            width: 100%;
            margin-top: 6px;
        }
        .timeline-slider {
            flex: 1;
            accent-color: var(--accent-cyan);
            cursor: pointer;
        }

        /* Two Columns for AI Inspection */
        .ai-inspect-layout {
            display: grid;
            grid-template-columns: 1fr 340px;
            gap: 16px;
        }
        @media (max-width: 900px) {
            .ai-inspect-layout { grid-template-columns: 1fr; }
        }

        /* Algo Tabs */
        .algo-tabs {
            display: flex;
            background-color: #0f172a;
            border-radius: 8px;
            padding: 4px;
            border: 1px solid #1e293b;
            gap: 4px;
        }
        .algo-tab {
            flex: 1;
            padding: 8px 4px;
            background-color: transparent;
            border: none;
            border-radius: 6px;
            color: var(--text-muted);
            font-size: 13px;
            font-weight: 700;
            cursor: pointer;
            text-align: center;
            transition: all 0.15s;
        }
        .algo-tab.active {
            background-color: var(--accent-blue);
            color: white;
        }

        /* Steps List */
        .steps-list-box {
            max-height: 180px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 4px;
            padding-right: 4px;
        }
        .step-row-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background-color: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 6px;
            padding: 6px 10px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .step-row-item:hover {
            border-color: var(--accent-cyan);
        }
        .step-row-item.active {
            border-color: var(--accent-cyan);
            background-color: rgba(56, 189, 248, 0.12);
        }

        /* OPEN and CLOSED Panels */
        .open-closed-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        .oc-box {
            background-color: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 10px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .oc-title {
            font-size: 11px;
            font-weight: 700;
            color: var(--text-muted);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .oc-list {
            max-height: 80px;
            overflow-y: auto;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: var(--accent-cyan);
            display: flex;
            flex-direction: column;
            gap: 2px;
        }

        /* Comparison Table */
        .table-responsive {
            overflow-x: auto;
            border-radius: 10px;
            border: 1px solid var(--border-color);
            background-color: #0c1322;
        }
        table.comp-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            white-space: nowrap;
        }
        table.comp-table th, table.comp-table td {
            padding: 10px 14px;
            border-bottom: 1px solid var(--border-color);
            vertical-align: middle;
        }
        table.comp-table th {
            background-color: #0f172a;
            color: var(--text-muted);
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.6px;
        }
        table.comp-table tbody tr {
            cursor: pointer;
            transition: background-color 0.15s ease;
        }
        table.comp-table tbody tr:hover {
            background-color: rgba(30, 41, 59, 0.45);
        }
        table.comp-table tbody tr.active-row {
            background-color: rgba(56, 189, 248, 0.09);
        }
        table.comp-table tbody tr.active-row td:first-child {
            box-shadow: inset 4px 0 0 var(--accent-cyan);
        }

        .algo-tag {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 44px;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 800;
            text-transform: uppercase;
        }
        .algo-tag-bfs { background: rgba(2, 132, 199, 0.18); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.35); }
        .algo-tag-ucs { background: rgba(20, 184, 166, 0.18); color: #2dd4bf; border: 1px solid rgba(45, 212, 191, 0.35); }
        .algo-tag-astar { background: rgba(168, 85, 247, 0.18); color: #c084fc; border: 1px solid rgba(192, 132, 252, 0.35); }

        /* Victory Popup Modal */
        .modal-backdrop {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background-color: rgba(0, 0, 0, 0.75);
            backdrop-filter: blur(4px);
            z-index: 1000;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .modal-card {
            background-color: #111827;
            border: 2px solid var(--accent-green);
            border-radius: 16px;
            padding: 28px;
            max-width: 460px;
            width: 100%;
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 16px;
            box-shadow: 0 10px 40px rgba(16, 185, 129, 0.3);
            animation: popIn 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }
        @keyframes popIn {
            from { transform: scale(0.85); opacity: 0; }
            to { transform: scale(1); opacity: 1; }
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-4px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

    <div class="container">
        <!-- HEADER -->
        <header>
            <div class="header-badge">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
                ETS Konsep Kecerdasan Artifisial
            </div>
            <h1>
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2.2"><path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"/></svg>
                Water Jug Puzzle & AI Search Visualizer
            </h1>
            <p class="subtitle">
                Visualisasi water jug menggunakan BFS, UCS, dan A* Search
            </p>
        </header>

        <!-- VISUALISASI AI ACTIVE BANNER -->
        <div class="ai-mode-header">
            <div class="ai-mode-pill">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a4 4 0 0 1 4 4v2h2a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2V6a4 4 0 0 1 4-4z"/></svg>
                <span>Visualisasi AI (BFS / UCS / A* Search)</span>
            </div>
        </div>

        <!-- PRESETS CARD -->
        <div class="solid-card">
            <!-- PRESETS BAR -->
            <div class="presets-bar" id="presets-bar">
                <span style="font-size:12px; font-weight:700; color:var(--text-muted); margin-right:4px;">Pilih Kasus Preset:</span>
                <button class="preset-chip highlight" id="preset-btn-video" onclick="applyPreset('video')">
                    <span class="chip-badge">3 Teko</span> 8-5-3 (Target 4L)
                </button>
                <button class="preset-chip" id="preset-btn-contrast" onclick="applyPreset('contrast')">
                    <span class="chip-badge">3 Teko</span> 10-9-1 (Target 5L)
                </button>
                <button class="preset-chip" id="preset-btn-4jugs" onclick="applyPreset('4jugs')">
                    <span class="chip-badge">4 Teko</span> 12-8-5-3 (Target 6L)
                </button>
                <button class="preset-chip" id="preset-btn-5jugs" onclick="applyPreset('5jugs')">
                    <span class="chip-badge">5 Teko</span> 16-9-7-5-2 (Target 8L)
                </button>
                <button class="preset-chip" id="preset-btn-custom" onclick="applyPreset('custom')">
                    <span class="chip-badge" style="background:rgba(245, 158, 11, 0.2); color:#fbbf24; border-color:rgba(245, 158, 11, 0.4);">✨ Kustom</span> Atur Sendiri (2–5 Teko)
                </button>
            </div>

            <!-- PRESET DESCRIPTION CARD (RINGKAS & JELAS) -->
            <div class="preset-info-card" id="preset-info-card">
                <div class="preset-info-header">
                    <span class="preset-info-tag" id="preset-info-tag">Kasus Klasik 3 Teko</span>
                    <span class="preset-info-title" id="preset-info-title">Teko 8-5-3 &rarr; Ukur 4 Liter</span>
                </div>
                <p class="preset-info-desc" id="preset-info-desc">
                    Kasus klasik teka-teki pembagian air (8, 5, 3 L) dengan target 4 Liter. Menguji BFS (langkah terpendek) dan A*.
                </p>
            </div>

            <!-- CUSTOM PRESET CONFIG PANEL -->
            <div id="custom-config-panel" style="display:none; margin-top:12px; padding:14px 16px; background-color:#0b1120; border:1px solid #1e293b; border-left:3px solid var(--accent-cyan); border-radius:8px;">
                <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px; margin-bottom:12px;">
                    <div style="display:flex; align-items:center; gap:8px;">
                        <span style="font-size:10px; font-weight:800; text-transform:uppercase; letter-spacing:0.5px; padding:2px 8px; border-radius:4px; background:rgba(56,189,248,0.15); color:var(--accent-cyan); border:1px solid rgba(56,189,248,0.3);">Preset Kustom</span>
                        <span style="font-size:13px; font-weight:700; color:var(--text-main);">Konfigurasi Teko & Goal State</span>
                    </div>
                    <div style="display:flex; align-items:center; gap:8px;">
                        <label style="font-size:12px; font-weight:700; color:var(--text-muted);">Jumlah Teko (2 - 5):</label>
                        <select id="custom-num-jugs" onchange="onCustomNumJugsChange(this.value)" style="background:#1e293b; border:1px solid #334155; color:var(--text-main); border-radius:6px; padding:4px 10px; font-size:12px; font-weight:700; outline:none; cursor:pointer;">
                            <option value="2">2 Teko</option>
                            <option value="3" selected>3 Teko</option>
                            <option value="4">4 Teko</option>
                            <option value="5">5 Teko</option>
                        </select>
                    </div>
                </div>

                <!-- Dynamic Jug Capacities Inputs -->
                <div style="margin-bottom:12px;">
                    <label style="display:block; font-size:12px; font-weight:700; color:var(--text-muted); margin-bottom:6px;">Kapasitas Tiap Teko (Liter):</label>
                    <div id="custom-caps-container" style="display:flex; gap:10px; flex-wrap:wrap;">
                        <!-- Rendered by JS -->
                    </div>
                </div>

                <!-- Single Goal State Input & Submit Button -->
                <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px; padding-top:10px; border-top:1px solid #1e293b;">
                    <div style="display:flex; align-items:center; gap:10px;">
                        <label style="font-size:12px; font-weight:700; color:var(--accent-cyan);">Target Goal State (1 Nilai):</label>
                        <div style="display:flex; align-items:center; gap:6px;">
                            <input type="number" id="custom-target-input" min="1" value="4" style="width:70px; background:#1e293b; border:1px solid var(--accent-cyan); border-radius:6px; padding:5px 8px; color:var(--accent-cyan); font-weight:800; font-family:'JetBrains Mono', monospace; text-align:center; font-size:14px; outline:none;">
                            <span style="font-size:12px; color:var(--text-muted); font-weight:600;">Liter (pada salah satu teko)</span>
                        </div>
                    </div>
                    <button class="btn-action primary" onclick="submitCustomPreset()" style="padding:6px 16px; font-size:12px;">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                        Hitung Solusi
                    </button>
                </div>
                <div id="custom-error-msg" style="display:none; margin-top:8px; font-size:12px; color:#fda4af; background:rgba(244,63,94,0.15); border:1px solid var(--accent-rose); border-radius:6px; padding:6px 10px;"></div>
            </div>
        </div>

        <!-- MAIN SIMULATION STAGE -->
        <div class="stage-card">
            <!-- STAGE HUD -->
            <div class="stage-hud">
                <div class="hud-status" id="hud-status-text">
                    <span style="color:var(--accent-cyan); font-weight:800;">Target:</span>
                    <span id="hud-target-desc">Ukur tepat 4 Liter pada salah satu teko</span>
                </div>
                <div class="hud-stats">
                    <div id="hud-ai-stats" style="display:flex; align-items:center; gap:14px;">
                        <span>Algoritma Aktif: <b class="stat-badge" id="hud-active-algo-name">BFS</b></span>
                        <span id="hud-initial-info" style="font-size:12px; color:var(--text-muted);">Air Awal: <b style="color:var(--accent-cyan);" id="hud-start-summary">Semua Teko Kosong (0L)</b></span>
                    </div>
                </div>
            </div>

            <!-- STREAM / POURING ACTION INDICATOR -->
            <div class="pouring-stream-indicator" id="pouring-indicator">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="7 10 12 15 17 10"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                <span id="pouring-indicator-text">Menuang air...</span>
            </div>

            <!-- JUGS GRAPHICS ROW -->
            <div class="jugs-row" id="jugs-render-area">
                <!-- Dynamically populated -->
            </div>

            <!-- AI STEP NARRATION & PLAYBACK CONTROLS (ALWAYS VISIBLE) -->
            <div id="ai-controls-section" style="display:flex; flex-direction:column; gap:12px;">
                <!-- STEP NARRATION CARD -->
                <div class="narration-card">
                    <div class="narration-header">
                        <span class="badge-step-pill" id="step-badge">LANGKAH 0</span>
                        <span class="narration-text" id="step-action-desc">Kondisi Awal</span>
                    </div>
                    <div class="node-eval-row">
                        <span>State: <b class="eval-val" id="eval-state">(0, 0, 0)</b></span>
                        <span>g(n) Liter: <b class="eval-val" id="eval-g">0 L</b></span>
                        <span>h(n) Estimasi: <b class="eval-val" id="eval-h">0</b></span>
                        <span>f(n) Total Cost: <b class="eval-val" id="eval-f">0</b></span>
                    </div>
                </div>

                <!-- PLAYBACK CONTROLS -->
                <div class="ai-playback-toolbar">
                    <button class="btn-action" onclick="jumpStep(0)">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="19 20 9 12 19 4 19 20"/><line x1="5" y1="19" x2="5" y2="5"/></svg>
                        Awal
                    </button>
                    <button class="btn-action" onclick="prevStep()">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="19 20 9 12 19 4 19 20"/></svg>
                        Prev
                    </button>
                    <button class="btn-action primary" id="btn-play" onclick="togglePlay()">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                        Play
                    </button>
                    <button class="btn-action" onclick="nextStep()">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 4 15 12 5 20 5 4"/></svg>
                        Next
                    </button>
                    <button class="btn-action" onclick="jumpEnd()">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 4 15 12 5 20 5 4"/><line x1="19" y1="5" x2="19" y2="19"/></svg>
                        Akhir
                    </button>

                    <div style="display:flex; align-items:center; gap:6px; margin-left:8px;">
                        <span style="font-size:12px; color:var(--text-muted); font-weight:600;">Kecepatan:</span>
                        <select id="sel-speed" onchange="updateSpeed(this.value)" style="background:#1e293b; border:1px solid #334155; color:white; border-radius:6px; padding:5px 8px; font-size:12px; outline:none;">
                            <option value="1600">Lambat (0.6x)</option>
                            <option value="900" selected>Normal (1.0x)</option>
                            <option value="450">Cepat (2.0x)</option>
                        </select>
                    </div>
                </div>

                <div class="timeline-slider-row">
                    <span style="font-size:12px; font-weight:700; color:var(--text-muted);" id="lbl-curr-step">0</span>
                    <input type="range" class="timeline-slider" id="scrubber" min="0" max="0" value="0" oninput="scrub(this.value)">
                    <span style="font-size:12px; font-weight:700; color:var(--text-muted);" id="lbl-max-step">0</span>
                </div>
            </div>
        </div>

        <!-- AI INSPECT SECTION (ALGO TABS, SOLUTION PATH, OPEN/CLOSED LISTS) -->
        <div id="ai-inspect-section" class="ai-inspect-layout" style="display:grid;">
            <!-- LEFT: COMPARISON TABLE -->
            <div class="solid-card" style="display:flex; flex-direction:column; gap:12px;">
                <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:8px;">
                    <h3 style="margin: 0; font-size:15px; color:var(--text-main); display:flex; align-items:center; gap:8px;">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2"><path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/></svg>
                        Tabel Perbandingan Metrik (BFS vs UCS vs A*)
                    </h3>
                    <span style="font-size:12px; color:var(--text-muted);">
                        Klik baris tabel untuk meninjau simulasi algoritma
                    </span>
                </div>
                <div class="table-responsive">
                    <table class="comp-table">
                        <thead>
                            <tr>
                                <th style="text-align:left;">Algoritma</th>
                                <th style="text-align:center;">Langkah</th>
                                <th style="text-align:center;">Cost Liter</th>
                                <th style="text-align:center;">Expanded</th>
                                <th style="text-align:center;">Generated</th>
                                <th style="text-align:center;">Maks OPEN</th>
                                <th style="text-align:center;">Waktu</th>
                                <th style="text-align:left;">Optimalitas</th>
                            </tr>
                        </thead>
                        <tbody id="comparison-tbody">
                            <tr><td colspan="8" style="text-align:center; color:var(--text-muted); padding:16px;">Memuat data...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- RIGHT: ALGO SELECTOR & OPEN/CLOSED LISTS -->
            <div class="solid-card" style="display:flex; flex-direction:column; gap:12px;">
                <div style="font-size:12px; font-weight:700; color:var(--text-muted); text-transform:uppercase;">
                    Algoritma yang Ditampilkan:
                </div>
                <div class="algo-tabs">
                    <button class="algo-tab active" id="tab-bfs" onclick="selectAlgoTab('bfs')">BFS</button>
                    <button class="algo-tab" id="tab-ucs" onclick="selectAlgoTab('ucs')">UCS</button>
                    <button class="algo-tab" id="tab-astar" onclick="selectAlgoTab('astar')">A*</button>
                </div>

                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:12px; font-weight:700; color:var(--accent-cyan);">Daftar Urutan Langkah:</span>
                    <span id="lbl-step-count" style="font-size:11px; color:var(--text-muted); font-weight:600;">0 Langkah</span>
                </div>

                <div class="steps-list-box" id="steps-container">
                    <!-- Populated dynamically -->
                </div>

                <!-- OPEN & CLOSED LISTS PREVIEW -->
                <div class="open-closed-container">
                    <div class="oc-box">
                        <div class="oc-title">
                            <span>OPEN (Frontier)</span>
                            <span id="cnt-open" class="mono" style="color:var(--accent-cyan);">0</span>
                        </div>
                        <div class="oc-list" id="list-open">
                            <span>-</span>
                        </div>
                    </div>
                    <div class="oc-box">
                        <div class="oc-title">
                            <span>CLOSED (Visited)</span>
                            <span id="cnt-closed" class="mono" style="color:#a78bfa;">0</span>
                        </div>
                        <div class="oc-list" id="list-closed">
                            <span>-</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

    </div>

<script>
    // State Aplikasi
    let currentMainMode = 'ai';
    let targetMode = 'single';
    let numJugs = 3;
    let solversData = null;

    // State AI Playback
    let activeAlgo = 'bfs';
    let currentStep = 0;
    let isPlaying = false;
    let timer = null;
    let playSpeed = 900;
    let debounceTimer = null;

    // --- DEFINISI PRESET TEMPLATE (RINGKAS DENGAN DESKRIPSI JELAS) ---
    const PRESETS = {
        'video': {
            name: '8-5-3 (Target 4L)',
            title: 'Teko 8-5-3 &rarr; Ukur 4 Liter',
            tag: 'Kasus Klasik 3 Teko',
            desc: 'Kasus klasik teka-teki pembagian air (8, 5, 3 L) dengan target 4 Liter. Mulai dari semua teko kosong, algoritma mengeksplorasi aksi isi penuh, buang, dan tuang antar teko.',
            numJugs: 3,
            caps: [8, 5, 3],
            targetAmt: 4
        },
        'contrast': {
            name: '10-9-1 (Target 5L)',
            title: 'Teko 10-9-1 &rarr; Ukur 5 Liter',
            tag: 'Kontras BFS vs UCS',
            desc: 'Kontras BFS vs UCS (10, 9, 1 L) target 5 Liter: BFS mengoptimalkan jumlah langkah aksi, sedangkan UCS dan A* mencari rute paling hemat volume liter tuang.',
            numJugs: 3,
            caps: [10, 9, 1],
            targetAmt: 5
        },
        '4jugs': {
            name: '12-8-5-3 (Target 6L)',
            title: 'Teko 12-8-5-3 &rarr; Ukur 6 Liter',
            tag: 'Kompleks 4 Teko',
            desc: 'Simulasi 4 Teko (12, 8, 5, 3 L) dengan target 6 Liter. Menunjukkan efisiensi penelusuran ruang keadaan multi-teko oleh BFS, UCS, dan A*.',
            numJugs: 4,
            caps: [12, 8, 5, 3],
            targetAmt: 6
        },
        '5jugs': {
            name: '16-9-7-5-2 (Target 8L)',
            title: 'Teko 16-9-7-5-2 &rarr; Ukur 8 Liter',
            tag: 'Tantangan 5 Teko',
            desc: 'Simulasi 5 Teko (16, 9, 7, 5, 2 L) dengan target 8 Liter. Menguji skalabilitas algoritma pencarian pada ruang state multi-teko.',
            numJugs: 5,
            caps: [16, 9, 7, 5, 2],
            targetAmt: 8
        },
        'custom': {
            name: 'Kustom (2-5 Teko)',
            title: 'Preset Kustom &rarr; Bebas Tentukan Teko & Target',
            tag: 'Kustomisasi Pengguna',
            desc: 'Pilih jumlah teko (2 sampai 5), tentukan kapasitas masing-masing teko, dan satu target volume air yang ingin diukur.',
            numJugs: 3,
            caps: [8, 5, 3],
            targetAmt: 4
        }
    };

    let activePresetKey = 'video';
    let solverAbortController = null;

    // State untuk Form Kustom
    let customJugsCount = 3;
    let customCaps = [8, 5, 3];
    let customTarget = 4;

    function onCustomNumJugsChange(val) {
        customJugsCount = parseInt(val);
        const defaultCaps = [8, 5, 3, 2, 1];
        while (customCaps.length < customJugsCount) {
            customCaps.push(defaultCaps[customCaps.length] || 2);
        }
        if (customCaps.length > customJugsCount) {
            customCaps = customCaps.slice(0, customJugsCount);
        }
        renderCustomCapsInputs();
    }

    function renderCustomCapsInputs() {
        const container = document.getElementById('custom-caps-container');
        if (!container) return;
        container.innerHTML = '';
        for (let i = 0; i < customJugsCount; i++) {
            const val = customCaps[i] || (i === 0 ? 8 : (i === 1 ? 5 : 3));
            const div = document.createElement('div');
            div.style.cssText = 'display:flex; flex-direction:column; gap:4px; background:#0f172a; border:1px solid #1e293b; border-radius:8px; padding:8px 12px; min-width:110px;';
            div.innerHTML = `
                <span style="font-size:11px; font-weight:700; color:var(--accent-cyan);">Teko ${i + 1}</span>
                <div style="display:flex; align-items:center; gap:6px;">
                    <input type="number" min="1" max="99" id="cap-input-${i}" value="${val}" onchange="customCaps[${i}] = parseInt(this.value) || 1" style="width:55px; background:#1e293b; border:1px solid #334155; border-radius:6px; padding:4px 6px; color:white; font-weight:700; font-family:'JetBrains Mono', monospace; text-align:center; font-size:13px; outline:none;">
                    <span style="font-size:11px; color:var(--text-muted); font-weight:600;">L</span>
                </div>
            `;
            container.appendChild(div);
        }
    }

    function submitCustomPreset() {
        const errEl = document.getElementById('custom-error-msg');
        if (errEl) errEl.style.display = 'none';

        const caps = [];
        for (let i = 0; i < customJugsCount; i++) {
            const input = document.getElementById(`cap-input-${i}`);
            const val = input ? parseInt(input.value) : 0;
            if (isNaN(val) || val <= 0) {
                if (errEl) {
                    errEl.textContent = `Kapasitas Teko ${i + 1} harus bilangan bulat positif (> 0)!`;
                    errEl.style.display = 'block';
                }
                return;
            }
            caps.push(val);
        }

        const targetInput = document.getElementById('custom-target-input');
        const targetAmt = targetInput ? parseInt(targetInput.value) : 0;
        if (isNaN(targetAmt) || targetAmt <= 0) {
            if (errEl) {
                errEl.textContent = 'Target volume harus bilangan bulat positif (> 0)!';
                errEl.style.display = 'block';
            }
            return;
        }

        const maxCap = Math.max(...caps);
        if (targetAmt > maxCap) {
            if (errEl) {
                errEl.textContent = `Target (${targetAmt}L) tidak boleh melebihi kapasitas teko terbesar (${maxCap}L)!`;
                errEl.style.display = 'block';
            }
            return;
        }

        // GCD feasibility check
        function gcd(a, b) { return b === 0 ? a : gcd(b, a % b); }
        let overallGcd = caps[0];
        for (let i = 1; i < caps.length; i++) {
            overallGcd = gcd(overallGcd, caps[i]);
        }
        if (targetAmt % overallGcd !== 0) {
            if (errEl) {
                errEl.textContent = `Target (${targetAmt}L) tidak mungkin dicapai karena bukan kelipatan FPB kapasitas teko (FPB = ${overallGcd})!`;
                errEl.style.display = 'block';
            }
            return;
        }

        customCaps = caps;
        customTarget = targetAmt;

        PRESETS['custom'].caps = caps;
        PRESETS['custom'].numJugs = caps.length;
        PRESETS['custom'].targetAmt = targetAmt;

        applyPreset('custom');
    }

    function setActivePresetUI(presetKey) {
        activePresetKey = presetKey;
        const keys = ['video', 'contrast', '4jugs', '5jugs', 'custom'];
        keys.forEach(k => {
            const btn = document.getElementById(`preset-btn-${k}`);
            if (btn) {
                if (k === presetKey) {
                    btn.classList.add('highlight');
                } else {
                    btn.classList.remove('highlight');
                }
            }
        });

        const infoCard = document.getElementById('preset-info-card');
        const customPanel = document.getElementById('custom-config-panel');

        if (presetKey === 'custom') {
            if (infoCard) infoCard.style.display = 'none';
            if (customPanel) {
                customPanel.style.display = 'block';
                renderCustomCapsInputs();
            }
        } else {
            if (customPanel) customPanel.style.display = 'none';
            if (infoCard) {
                infoCard.style.display = 'flex';
                const tagEl = document.getElementById('preset-info-tag');
                const titleEl = document.getElementById('preset-info-title');
                const descEl = document.getElementById('preset-info-desc');

                if (PRESETS[presetKey]) {
                    const p = PRESETS[presetKey];
                    if (tagEl) {
                        tagEl.textContent = p.tag;
                        tagEl.style.color = '#fbbf24';
                        tagEl.style.borderColor = 'rgba(245, 158, 11, 0.3)';
                        tagEl.style.background = 'rgba(245, 158, 11, 0.15)';
                    }
                    if (titleEl) titleEl.innerHTML = p.title;
                    if (descEl) descEl.textContent = p.desc;
                }
            }
        }
    }

    function applyPreset(key) {
        const p = PRESETS[key];
        if (!p) return;

        setActivePresetUI(key);
        stopPlay();

        numJugs = p.numJugs;

        const hudTarget = document.getElementById('hud-target-desc');
        if (hudTarget) hudTarget.textContent = `Ukur tepat ${p.targetAmt} Liter pada salah satu teko`;

        const hudSummary = document.getElementById('hud-start-summary');
        if (hudSummary) hudSummary.textContent = `Semua Teko Kosong (0L)`;

        currentStep = 0;
        const initialStart = new Array(p.caps.length).fill(0);
        renderAIJugsStage(initialStart, p.caps, null, null, 'start');
        const lblCurrStep = document.getElementById('lbl-curr-step');
        if (lblCurrStep) lblCurrStep.textContent = '0';
        const scrubber = document.getElementById('scrubber');
        if (scrubber) scrubber.value = 0;

        runSolversForPreset(p);
    }

    // --- CALL BACKEND SOLVER FOR ACTIVE PRESET ---
    async function runSolversForPreset(p) {
        if (solverAbortController) {
            solverAbortController.abort();
        }
        solverAbortController = new AbortController();

        try {
            const res = await fetch('/solve', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                signal: solverAbortController.signal,
                body: JSON.stringify({
                    caps: p.caps,
                    target_amount: p.targetAmt
                })
            });
            const data = await res.json();

            if (!data.valid) {
                const errBanner = document.getElementById('custom-error-msg');
                if (errBanner) {
                    errBanner.textContent = data.message;
                    errBanner.style.display = 'block';
                }
                return;
            }

            solversData = data;
            renderComparisonTable(data);
            selectAlgoTab(activeAlgo);

        } catch (err) {
            if (err.name !== 'AbortError') {
                console.error(err);
            }
        }
    }

    function showPourIndicator(text) {
        const el = document.getElementById('pouring-indicator');
        document.getElementById('pouring-indicator-text').textContent = text;
        el.classList.add('active');
        clearTimeout(el._timer);
        el._timer = setTimeout(() => {
            el.classList.remove('active');
        }, 3500);
    }

    function hidePourIndicator() {
        document.getElementById('pouring-indicator').classList.remove('active');
    }

    // Helper untuk ukuran relatif teko (Scaling per 2 L)
    function getJugDimensions(cap) {
        const height = Math.round(90 + (cap / 2.0) * 30);
        const width = Math.round(76 + Math.min(36, (cap / 2.0) * 6));
        return { height, width };
    }

    // Helper untuk skala garis ukur (HANYA teks kelipatan 2 saja)
    function getMeasuringScaleHtml(cap) {
        let ticksHtml = '';
        for (let t = 1; t <= cap; t++) {
            const bPct = (t / cap) * 100;
            const isStep2L = (t % 2 === 0);

            if (isStep2L) {
                let labelStyle = `bottom:${bPct}%;`;
                if (bPct >= 90) {
                    labelStyle = 'top: 3px; transform: none;';
                } else if (bPct <= 12) {
                    labelStyle = 'bottom: 3px; transform: none;';
                } else {
                    labelStyle = `bottom:${bPct}%; transform: translateY(50%);`;
                }

                ticksHtml += `
                    <div class="scale-tick scale-tick-major" style="bottom:${Math.min(99, bPct)}%;"></div>
                    <span class="scale-label" style="${labelStyle}">${t}L</span>
                `;
            } else {
                ticksHtml += `
                    <div class="scale-tick scale-tick-minor" style="bottom:${Math.min(99, bPct)}%;"></div>
                `;
            }
        }
        return ticksHtml;
    }

    // ==============================================================================
    // SIMULASI AI SEARCH (BFS, UCS, A*)
    // ==============================================================================
    function renderComparisonTable(data) {
        const tbody = document.getElementById('comparison-tbody');
        tbody.innerHTML = '';

        const algos = [
            { key: 'bfs', tag: 'BFS', fullName: 'Breadth-First Search', tagClass: 'algo-tag-bfs', desc: 'Jumlah langkah paling sedikit' },
            { key: 'ucs', tag: 'UCS', fullName: 'Uniform-Cost Search', tagClass: 'algo-tag-ucs', desc: 'Total liter dipindah paling sedikit' },
            { key: 'astar', tag: 'A*', fullName: 'A* Search', tagClass: 'algo-tag-astar', desc: 'Optimal liter + estimasi heuristik' }
        ];

        algos.forEach(a => {
            const res = data[a.key];
            const isRowActive = a.key === activeAlgo ? 'class="active-row"' : '';
            if (!res || !res.found) {
                tbody.innerHTML += `
                    <tr ${isRowActive} onclick="selectAlgoTab('${a.key}')">
                        <td>
                            <div style="display:flex; align-items:center; gap:8px;">
                                <span class="algo-tag ${a.tagClass}">${a.tag}</span>
                                <b>${a.fullName}</b>
                            </div>
                        </td>
                        <td colspan="6" style="text-align:center; color:var(--accent-rose); font-weight:700;">Tidak ada solusi</td>
                        <td><span style="font-size:11px; color:var(--text-muted);">${a.desc}</span></td>
                    </tr>
                `;
            } else {
                const timeFormatted = typeof res.time_ms === 'number' ? res.time_ms.toFixed(3) : res.time_ms;
                tbody.innerHTML += `
                    <tr ${isRowActive} onclick="selectAlgoTab('${a.key}')">
                        <td>
                            <div style="display:flex; align-items:center; gap:8px;">
                                <span class="algo-tag ${a.tagClass}">${a.tag}</span>
                                <b style="font-size:12px;">${a.fullName}</b>
                            </div>
                        </td>
                        <td style="text-align:center;"><b class="mono">${res.steps}</b> lkg</td>
                        <td style="text-align:center;"><b class="mono" style="color:var(--accent-cyan);">${res.total_liters}</b> L</td>
                        <td style="text-align:center;" class="mono">${res.expanded}</td>
                        <td style="text-align:center;" class="mono">${res.generated}</td>
                        <td style="text-align:center;" class="mono">${res.max_open}</td>
                        <td style="text-align:center;"><span class="mono">${timeFormatted}</span> ms</td>
                        <td style="font-size:11px; color:var(--accent-green);">&#10003; ${a.desc}</td>
                    </tr>
                `;
            }
        });
    }

    function selectAlgoTab(algoKey) {
        activeAlgo = algoKey;
        document.querySelectorAll('.algo-tab').forEach(t => t.classList.remove('active'));
        const tabEl = document.getElementById(`tab-${algoKey}`);
        if (tabEl) tabEl.classList.add('active');

        document.getElementById('hud-active-algo-name').textContent = algoKey.toUpperCase();

        if (!solversData) return;
        renderComparisonTable(solversData);

        const chosen = solversData[algoKey];
        if (!chosen) return;
        const path = chosen.path || [];

        document.getElementById('lbl-step-count').textContent = chosen.found ? `${chosen.steps} Langkah (${chosen.total_liters} L)` : 'Tidak ada solusi';

        const scrubber = document.getElementById('scrubber');
        scrubber.max = Math.max(0, path.length - 1);
        document.getElementById('lbl-max-step').textContent = scrubber.max;

        renderSolutionStepsList(path);
        renderOpenClosed(chosen);

        if (!chosen.found || path.length === 0) {
            stopPlay();
            currentStep = 0;
            const p = PRESETS[activePresetKey] || PRESETS['video'];
            const caps = (solversData && solversData.caps) || p.caps;
            const start = (solversData && solversData.start) || new Array(caps.length).fill(0);
            renderAIJugsStage(start, caps, null, null, 'start');

            const badge = document.getElementById('step-badge');
            badge.textContent = 'TIDAK ADA SOLUSI';
            badge.style.backgroundColor = 'rgba(244, 63, 94, 0.2)';
            badge.style.color = 'var(--accent-rose)';

            document.getElementById('step-action-desc').textContent = 'Target tidak dapat dicapai dari kondisi air awal ini.';
            document.getElementById('eval-state').textContent = `(${start.join(', ')})`;
            document.getElementById('eval-g').textContent = '-';
            document.getElementById('eval-h').textContent = '-';
            document.getElementById('eval-f').textContent = '-';

            document.getElementById('lbl-curr-step').textContent = '0';
            document.getElementById('lbl-max-step').textContent = '0';
            scrubber.value = 0;
            scrubber.disabled = true;
            return;
        }

        scrubber.disabled = false;
        jumpStep(0);
    }

    function renderSolutionStepsList(path) {
        const box = document.getElementById('steps-container');
        box.innerHTML = '';
        if (!path || path.length === 0) {
            box.innerHTML = '<div style="color:var(--accent-rose); text-align:center; padding:12px; font-weight:700;">Tidak ada solusi yang ditemukan.</div>';
            return;
        }

        path.forEach((step, idx) => {
            const row = document.createElement('div');
            row.id = `step-row-${idx}`;
            row.className = `step-row-item ${idx === currentStep ? 'active' : ''}`;
            row.onclick = () => jumpStep(idx);
            row.innerHTML = `
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="font-size:11px; font-weight:800; color:var(--text-muted); width:18px;">#${idx}</span>
                    <span style="white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:180px;">${step.action}</span>
                </div>
                <span class="mono" style="font-weight:700; color:var(--accent-cyan); font-size:11px;">(${step.state.join(', ')})</span>
            `;
            box.appendChild(row);
        });
    }

    function renderOpenClosed(algoData) {
        const openBox = document.getElementById('list-open');
        const closedBox = document.getElementById('list-closed');
        document.getElementById('cnt-open').textContent = algoData.max_open || 0;
        document.getElementById('cnt-closed').textContent = algoData.expanded || 0;

        openBox.innerHTML = (algoData.open_sample || []).map(s => `<div>(${s.join(',')})</div>`).join('') || '<div>-</div>';
        closedBox.innerHTML = (algoData.closed_sample || []).map(s => `<div>(${s.join(',')})</div>`).join('') || '<div>-</div>';
    }

    function renderAIJugsStage(state, caps, src, dst, actionType) {
        const stageArea = document.getElementById('jugs-render-area');
        stageArea.innerHTML = '';
        const p = PRESETS[activePresetKey] || PRESETS['video'];
        const targetAmount = (solversData && solversData.target_amount) ? solversData.target_amount : p.targetAmt;

        for (let i = 0; i < state.length; i++) {
            const vol = state[i];
            const cap = caps[i];
            const pct = Math.min(100, Math.round((vol / cap) * 100));
            const dim = getJugDimensions(cap);
            const ticksHtml = getMeasuringScaleHtml(cap);

            const isFill = (actionType === 'fill' && dst !== null && dst === i);
            const isEmpty = (actionType === 'empty' && src !== null && src === i);
            const isSource = (actionType === 'pour' && src !== null && src === i);
            const isTarget = (actionType === 'pour' && dst !== null && dst === i);
            const isTargetAchieved = (vol === targetAmount);

            const item = document.createElement('div');
            item.className = 'jug-item';
            if (isFill) item.classList.add('ai-fill');
            if (isEmpty) item.classList.add('ai-empty');
            if (isSource) item.classList.add('ai-source');
            if (isTarget) item.classList.add('ai-target');
            if (isTargetAchieved) item.classList.add('target-reached');

            let actionBadgeHtml = '';
            if (isTargetAchieved) {
                actionBadgeHtml = `<div class="jug-win-badge">Target (${vol}L)</div>`;
            } else if (isFill) {
                actionBadgeHtml = `<div class="jug-tag-badge" style="display:block; background:#0284c7; color:white;">💧 ISI PENUH</div>`;
            } else if (isEmpty) {
                actionBadgeHtml = `<div class="jug-tag-badge" style="display:block; background:#e11d48; color:white;">🚰 BUANG AIR</div>`;
            } else if (isSource) {
                actionBadgeHtml = `<div class="jug-tag-badge" style="display:block; background:var(--accent-amber); color:#111827;">SUMBER TUANG</div>`;
            } else if (isTarget) {
                actionBadgeHtml = `<div class="jug-tag-badge" style="display:block; background:var(--accent-green); color:white;">PENERIMA AIR</div>`;
            }

            item.innerHTML = `
                ${actionBadgeHtml}
                <div class="jug-header-label">
                    Teko ${i + 1} (${cap}L)
                </div>
                <div class="jug-glass-body" style="height:${dim.height}px; width:${dim.width}px;">
                    <div class="jug-measuring-scale">${ticksHtml}</div>
                    <div class="water-fill" style="height:${pct}%;">
                        <div class="water-wave-top"></div>
                    </div>
                </div>
                <div class="jug-badge-info" style="min-width:${dim.width}px;">${vol} / ${cap} L</div>
            `;
            stageArea.appendChild(item);
        }
    }

    function jumpStep(idx) {
        if (!solversData || !solversData[activeAlgo]) return;
        const chosen = solversData[activeAlgo];
        if (!chosen.found) return;
        const path = chosen.path;
        if (!path || path.length === 0) return;

        currentStep = Math.max(0, Math.min(idx, path.length - 1));
        const step = path[currentStep];
        const caps = solversData.caps;

        renderAIJugsStage(step.state, caps, step.src, step.dst, step.action_type);

        const badge = document.getElementById('step-badge');
        badge.textContent = `LANGKAH ${currentStep} / ${path.length - 1}`;
        badge.style.backgroundColor = 'rgba(56, 189, 248, 0.15)';
        badge.style.color = 'var(--accent-cyan)';

        document.getElementById('step-action-desc').textContent = step.action;
        document.getElementById('eval-state').textContent = `(${step.state.join(', ')})`;
        document.getElementById('eval-g').textContent = `${step.g} L`;
        document.getElementById('eval-h').textContent = `${step.h}`;
        document.getElementById('eval-f').textContent = `${step.f}`;

        document.getElementById('lbl-curr-step').textContent = currentStep;
        document.getElementById('scrubber').value = currentStep;

        if (step.action_type === 'fill') {
            showPourIndicator(`Mengisi penuh Teko ${(step.dst !== null ? step.dst + 1 : '')} (${caps[step.dst]}L) dari keran...`);
        } else if (step.action_type === 'empty') {
            showPourIndicator(`Membuang seluruh air dari Teko ${(step.src !== null ? step.src + 1 : '')}...`);
        } else if (step.action_type === 'pour') {
            showPourIndicator(`Menuang ${step.amount}L dari Teko ${(step.src !== null ? step.src + 1 : '')} ke Teko ${(step.dst !== null ? step.dst + 1 : '')}...`);
        } else {
            hidePourIndicator();
        }

        document.querySelectorAll('.step-row-item').forEach(r => r.classList.remove('active'));
        const activeRow = document.getElementById(`step-row-${currentStep}`);
        if (activeRow) {
            activeRow.classList.add('active');
            activeRow.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }

    function nextStep() {
        if (!solversData || !solversData[activeAlgo] || !solversData[activeAlgo].found) return;
        const path = solversData[activeAlgo].path;
        if (!path || currentStep >= path.length - 1) {
            stopPlay();
            return;
        }
        jumpStep(currentStep + 1);
    }

    function prevStep() {
        if (!solversData || !solversData[activeAlgo] || !solversData[activeAlgo].found) return;
        if (currentStep <= 0) return;
        jumpStep(currentStep - 1);
    }

    function jumpEnd() {
        if (!solversData || !solversData[activeAlgo] || !solversData[activeAlgo].found) return;
        const path = solversData[activeAlgo].path;
        if (!path) return;
        jumpStep(path.length - 1);
    }

    function scrub(val) {
        if (!solversData || !solversData[activeAlgo] || !solversData[activeAlgo].found) return;
        jumpStep(parseInt(val));
    }

    function togglePlay() {
        if (!solversData || !solversData[activeAlgo] || !solversData[activeAlgo].found) return;
        if (isPlaying) {
            stopPlay();
        } else {
            startPlay();
        }
    }

    function startPlay() {
        if (!solversData || !solversData[activeAlgo] || !solversData[activeAlgo].found) return;
        const path = solversData[activeAlgo].path;
        if (!path || path.length <= 1) return;

        if (currentStep >= path.length - 1) {
            jumpStep(0);
        }

        isPlaying = true;
        const btn = document.getElementById('btn-play');
        btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg> Pause`;
        btn.style.backgroundColor = '#e11d48';

        timer = setInterval(() => {
            if (currentStep >= path.length - 1) {
                stopPlay();
            } else {
                nextStep();
            }
        }, playSpeed);
    }

    function stopPlay() {
        isPlaying = false;
        clearInterval(timer);
        const btn = document.getElementById('btn-play');
        if (btn) {
            btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg> Play`;
            btn.style.backgroundColor = 'var(--accent-blue)';
        }
    }

    function updateSpeed(val) {
        playSpeed = parseInt(val);
        if (isPlaying) {
            clearInterval(timer);
            timer = setInterval(() => {
                if (currentStep >= solversData[activeAlgo].path.length - 1) {
                    stopPlay();
                } else {
                    nextStep();
                }
            }, playSpeed);
        }
    }

    // Inisialisasi awal dengan Preset Video 8-5-3 (Target 4L)
    applyPreset('video');
</script>
</body>
</html>
"""

# ==============================================================================
# BAGIAN 3: ROUTING FLASK
# ==============================================================================
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/solve", methods=["POST"])
def solve():
    req = request.get_json() or {}
    caps = [int(x) for x in req.get("caps", [8, 5, 3])]
    target_amount = int(req.get("target_amount", 4))
    start = [0] * len(caps)

    is_valid, msg = validate_inputs(caps, target_amount)
    if not is_valid:
        return jsonify({
            "valid": False,
            "message": msg
        })

    goal_checker = make_goal_checker(target_amount)
    heuristic_func = make_heuristic(target_amount)

    bfs_res = solve_bfs(caps, start, goal_checker, heuristic_func)
    ucs_res = solve_ucs(caps, start, goal_checker, heuristic_func)
    astar_res = solve_astar(caps, start, goal_checker, heuristic_func)

    return jsonify({
        "valid": True,
        "message": "Sukses",
        "caps": caps,
        "start": start,
        "target_amount": target_amount,
        "bfs": bfs_res,
        "ucs": ucs_res,
        "astar": astar_res
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)