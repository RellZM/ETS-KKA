from flask import Flask, render_template_string, request, jsonify
import math
import time
from collections import deque
import heapq

app = Flask(__name__)

# ==============================================================================
# BAGIAN 1: LOGIKA AI SEARCH - WATER JUG PROBLEM (BFS, UCS, A*)
# ==============================================================================
# Aturan Sistem:
# 1. State: Tuple isi tiap teko (v_1, v_2, ..., v_n)
# 2. Operator tunggal: Menuang dari Teko i ke Teko j (i != j)
# 3. Konservasi Air: Total air Start == Total air Goal (tidak ada keran / buang)
# 4. Cost:
#    - BFS: 1 langkah per tuangan (Optimalitas: Jumlah langkah paling sedikit)
#    - UCS: g(n) = akumulasi total liter yang dipindah (Optimalitas: Total liter paling sedikit)
#    - A* : f(n) = g(n) + h(n), h(n) = 0.5 * sum(|v_k - goal_k|) (Admissible heuristic)
# ==============================================================================

def heuristic(state, goal):
    """
    Heuristik Admisibel A*:
    h(n) = 0.5 * sum(|isi_k(n) - goal_k|)
    Setiap tuangan a liter paling banyak mengurangi selisih total sebesar 2a,
    sehingga h(n) tidak pernah melebihi cost sebenarnya (total liter yang harus dipindah).
    """
    return sum(abs(s - g) for s, g in zip(state, goal)) / 2.0

def get_neighbors(state, caps):
    """
    Membangkitkan semua transisi tuang yang valid dari teko i ke teko j.
    Jumlah kemungkinan aksi maksimal n * (n - 1).
    """
    n = len(state)
    neighbors = []
    for i in range(n):
        if state[i] == 0:
            continue
        for j in range(n):
            if i == j:
                continue
            if state[j] < caps[j]:
                amount = min(state[i], caps[j] - state[j])
                new_state = list(state)
                new_state[i] -= amount
                new_state[j] += amount
                action_text = f"Tuang {amount}L dari Teko {i+1} ke Teko {j+1}"
                neighbors.append((tuple(new_state), amount, action_text, i, j))
    return neighbors

def validate_inputs(caps, start, goal):
    """
    Validasi aturan input:
    - 2 <= jumlah teko <= 5
    - Kapasitas > 0
    - 0 <= start <= caps, 0 <= goal <= caps
    - Total start == Total goal
    """
    n = len(caps)
    if not (2 <= n <= 5):
        return False, f"Jumlah teko harus antara 2 sampai 5 teko (input: {n} teko)!"
    if len(start) != n or len(goal) != n:
        return False, "Panjang array kapasitas, start, dan goal harus sama!"
    
    for i in range(n):
        if caps[i] <= 0:
            return False, f"Kapasitas Teko {i+1} harus bilangan bulat positif (> 0)!"
        if not (0 <= start[i] <= caps[i]):
            return False, f"Isi start Teko {i+1} ({start[i]}L) harus antara 0 sampai kapasitas ({caps[i]}L)!"
        if not (0 <= goal[i] <= caps[i]):
            return False, f"Isi goal Teko {i+1} ({goal[i]}L) harus antara 0 sampai kapasitas ({caps[i]}L)!"
            
    sum_start = sum(start)
    sum_goal = sum(goal)
    if sum_start != sum_goal:
        return False, f"Goal {sum_goal} L, padahal air start {sum_start} L! Total air start dan goal harus sama karena aksi tuang tidak menambah/mengurangi air."
        
    return True, "Valid"

# 1. BREADTH-FIRST SEARCH (BFS)
def solve_bfs(caps, start, goal):
    t0 = time.perf_counter()
    start_t, goal_t = tuple(start), tuple(goal)
    
    if start_t == goal_t:
        return {
            "name": "BFS (Breadth-First Search)",
            "found": True,
            "path": [{"state": list(start_t), "action": "Kondisi Awal (Start = Goal)", "amount": 0, "g": 0, "h": 0, "f": 0, "src": None, "dst": None}],
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
    came_from = {start_t: (None, "Mulai (Kondisi Awal)", 0, None, None)}
    g_score = {start_t: 0}
    
    expanded = 0
    generated = 1
    max_open = 1
    found = False
    
    while queue:
        max_open = max(max_open, len(queue))
        curr = queue.popleft()
        expanded += 1
        
        if curr == goal_t:
            found = True
            break
            
        for nxt, amount, action, src, dst in get_neighbors(curr, caps):
            if nxt not in visited:
                visited.add(nxt)
                came_from[nxt] = (curr, action, amount, src, dst)
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
    curr = goal_t
    while curr is not None:
        prev, action, amount, src, dst = came_from[curr]
        h_val = heuristic(curr, goal_t)
        g_val = g_score[curr]
        path.append({
            "state": list(curr),
            "action": action,
            "amount": amount,
            "g": g_val,
            "h": h_val,
            "f": g_val + h_val,
            "src": src,
            "dst": dst
        })
        curr = prev
    path.reverse()
    
    return {
        "name": "BFS (Breadth-First Search)",
        "found": True,
        "path": path,
        "steps": max(0, len(path) - 1),
        "total_liters": g_score[goal_t],
        "expanded": expanded,
        "generated": generated,
        "max_open": max_open,
        "time_ms": time_ms,
        "open_sample": [list(s) for s in list(queue)[:10]],
        "closed_sample": [list(s) for s in list(visited)[:10]]
    }

# 2. UNIFORM-COST SEARCH (UCS)
def solve_ucs(caps, start, goal):
    t0 = time.perf_counter()
    start_t, goal_t = tuple(start), tuple(goal)
    
    if start_t == goal_t:
        return {
            "name": "UCS (Uniform-Cost Search)",
            "found": True,
            "path": [{"state": list(start_t), "action": "Kondisi Awal (Start = Goal)", "amount": 0, "g": 0, "h": 0, "f": 0, "src": None, "dst": None}],
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
    came_from = {start_t: (None, "Mulai (Kondisi Awal)", 0, None, None)}
    g_score = {start_t: 0}
    closed = set()
    
    expanded = 0
    generated = 1
    max_open = 1
    found = False
    
    while pq:
        max_open = max(max_open, len(pq))
        curr_g, _, curr = heapq.heappop(pq)
        
        if curr in closed:
            continue
        closed.add(curr)
        expanded += 1
        
        if curr == goal_t:
            found = True
            break
            
        for nxt, amount, action, src, dst in get_neighbors(curr, caps):
            tentative_g = curr_g + amount
            if nxt not in g_score or tentative_g < g_score[nxt]:
                g_score[nxt] = tentative_g
                came_from[nxt] = (curr, action, amount, src, dst)
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
    curr = goal_t
    while curr is not None:
        prev, action, amount, src, dst = came_from[curr]
        h_val = heuristic(curr, goal_t)
        g_val = g_score[curr]
        path.append({
            "state": list(curr),
            "action": action,
            "amount": amount,
            "g": g_val,
            "h": h_val,
            "f": g_val + h_val,
            "src": src,
            "dst": dst
        })
        curr = prev
    path.reverse()
    
    return {
        "name": "UCS (Uniform-Cost Search)",
        "found": True,
        "path": path,
        "steps": max(0, len(path) - 1),
        "total_liters": g_score[goal_t],
        "expanded": expanded,
        "generated": generated,
        "max_open": max_open,
        "time_ms": time_ms,
        "open_sample": [list(item[2]) for item in pq[:10]],
        "closed_sample": [list(s) for s in list(closed)[:10]]
    }

# 3. A* SEARCH (INFORMED SEARCH)
def solve_astar(caps, start, goal):
    t0 = time.perf_counter()
    start_t, goal_t = tuple(start), tuple(goal)
    
    if start_t == goal_t:
        return {
            "name": "A* Search",
            "found": True,
            "path": [{"state": list(start_t), "action": "Kondisi Awal (Start = Goal)", "amount": 0, "g": 0, "h": 0, "f": 0, "src": None, "dst": None}],
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
    h_start = heuristic(start_t, goal_t)
    pq = [(h_start, counter, start_t)]
    came_from = {start_t: (None, "Mulai (Kondisi Awal)", 0, None, None)}
    g_score = {start_t: 0}
    closed = set()
    
    expanded = 0
    generated = 1
    max_open = 1
    found = False
    
    while pq:
        max_open = max(max_open, len(pq))
        curr_f, _, curr = heapq.heappop(pq)
        
        if curr in closed:
            continue
        closed.add(curr)
        expanded += 1
        
        if curr == goal_t:
            found = True
            break
            
        for nxt, amount, action, src, dst in get_neighbors(curr, caps):
            tentative_g = g_score[curr] + amount
            if nxt not in g_score or tentative_g < g_score[nxt]:
                g_score[nxt] = tentative_g
                h_nxt = heuristic(nxt, goal_t)
                f_nxt = tentative_g + h_nxt
                came_from[nxt] = (curr, action, amount, src, dst)
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
    curr = goal_t
    while curr is not None:
        prev, action, amount, src, dst = came_from[curr]
        h_val = heuristic(curr, goal_t)
        g_val = g_score[curr]
        path.append({
            "state": list(curr),
            "action": action,
            "amount": amount,
            "g": g_val,
            "h": h_val,
            "f": g_val + h_val,
            "src": src,
            "dst": dst
        })
        curr = prev
    path.reverse()
    
    return {
        "name": "A* Search",
        "found": True,
        "path": path,
        "steps": max(0, len(path) - 1),
        "total_liters": g_score[goal_t],
        "expanded": expanded,
        "generated": generated,
        "max_open": max_open,
        "time_ms": time_ms,
        "open_sample": [list(item[2]) for item in pq[:10]],
        "closed_sample": [list(s) for s in list(closed)[:10]]
    }


# ==============================================================================
# BAGIAN 2: TEMPLATE HTML + CLEAN SOLID STYLING (NO GRADIENTS, NO EMOJIS)
# ==============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visualisasi Water Jug (BFS, UCS, A*)</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-page: #0b1120;
            --card-bg: #111827;
            --stage-bg: #0f172a;
            --border-color: #1e293b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-cyan: #38bdf8;
            --accent-blue: #0284c7;
            --accent-green: #10b981;
            --accent-amber: #f59e0b;
            --accent-rose: #f43f5e;
            --water-color: #0284c7;
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
            gap: 20px;
        }

        /* Header */
        header {
            text-align: center;
            padding: 8px 0;
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
            font-weight: 600;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }
        h1 {
            margin: 0 0 6px 0;
            font-size: 26px;
            font-weight: 800;
            color: var(--text-main);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        p.subtitle {
            margin: 0;
            color: var(--text-muted);
            font-size: 14px;
        }

        /* Solid Cards */
        .solid-card {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 20px;
        }

        /* Form & Configuration */
        .config-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 12px;
            margin-bottom: 16px;
        }
        .jug-count-selector {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .btn-jug-count {
            background-color: #1e293b;
            border: 1px solid #334155;
            color: var(--text-main);
            padding: 6px 14px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 700;
            transition: all 0.15s ease;
        }
        .btn-jug-count.active {
            background-color: var(--accent-blue);
            border-color: var(--accent-cyan);
            color: white;
        }

        /* Dynamic Table Grid for Inputs */
        .jugs-input-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 14px;
            margin-bottom: 16px;
        }
        .jug-input-card {
            background-color: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 10px;
            padding: 14px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .jug-input-header {
            font-size: 13px;
            font-weight: 700;
            color: var(--accent-cyan);
            display: flex;
            align-items: center;
            gap: 6px;
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
            width: 80px;
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 6px;
            padding: 6px 10px;
            color: var(--text-main);
            font-size: 14px;
            font-weight: 700;
            outline: none;
            font-family: 'JetBrains Mono', monospace;
            text-align: right;
        }
        .row-input input:focus {
            border-color: var(--accent-cyan);
        }

        /* Water Sum Validation Bar */
        .water-sum-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 10px;
            background-color: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 10px;
            padding: 10px 16px;
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

        /* Actions & Presets */
        .action-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 14px;
        }
        .presets-list {
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
            align-items: center;
        }
        .btn-preset {
            background-color: #1e293b;
            border: 1px solid #334155;
            color: var(--text-muted);
            padding: 6px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .btn-preset:hover {
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
        }

        .btn-solve-all {
            background-color: var(--accent-blue);
            border: 1px solid #38bdf8;
            color: white;
            font-weight: 700;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            transition: opacity 0.15s ease;
        }
        .btn-solve-all:hover {
            opacity: 0.9;
        }
        .btn-solve-all:disabled {
            background-color: #334155;
            border-color: #475569;
            color: #94a3b8;
            cursor: not-allowed;
        }

        /* Error Notification Banner */
        #error-banner {
            display: none;
            background-color: rgba(244, 63, 94, 0.15);
            border: 1px solid var(--accent-rose);
            color: #fda4af;
            padding: 12px 16px;
            border-radius: 10px;
            font-size: 13px;
            font-weight: 600;
            align-items: center;
            gap: 8px;
        }

        /* Stage Layout */
        .stage-layout {
            display: grid;
            grid-template-columns: 1fr 390px;
            gap: 20px;
        }
        @media (max-width: 950px) {
            .stage-layout { grid-template-columns: 1fr; }
        }

        /* Jugs Visualization Canvas */
        .jugs-stage {
            background-color: var(--stage-bg);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 24px 16px;
            min-height: 380px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: space-between;
            gap: 20px;
        }

        .jugs-row {
            display: flex;
            align-items: flex-end;
            justify-content: center;
            gap: 28px;
            width: 100%;
            flex-wrap: wrap;
            padding: 10px 0;
        }

        .jug-col {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
            position: relative;
        }
        .jug-col-title {
            font-size: 13px;
            font-weight: 700;
            color: var(--accent-cyan);
            display: flex;
            align-items: center;
            gap: 4px;
        }
        .jug-glass {
            position: relative;
            width: 90px;
            height: 190px;
            background-color: rgba(255, 255, 255, 0.02);
            border: 3px solid #64748b;
            border-top: none;
            border-radius: 0 0 16px 16px;
            overflow: hidden;
            display: flex;
            flex-direction: column-reverse;
            box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.5);
            transition: border-color 0.2s;
        }
        .jug-glass.source-pour {
            border-color: var(--accent-amber) !important;
        }
        .jug-glass.target-pour {
            border-color: var(--accent-green) !important;
        }

        .water-column {
            width: 100%;
            height: 0%;
            background-color: var(--water-color);
            transition: height 0.5s ease-in-out;
            position: relative;
        }
        .water-top-line {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 3px;
            background-color: #38bdf8;
        }

        .measuring-scale {
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 100%;
            pointer-events: none;
        }
        .scale-mark {
            position: absolute;
            left: 0;
            width: 12px;
            height: 1px;
            background-color: rgba(255, 255, 255, 0.35);
        }
        .scale-num {
            position: absolute;
            left: 16px;
            font-size: 9px;
            font-weight: 700;
            color: rgba(255, 255, 255, 0.5);
            font-family: 'JetBrains Mono', monospace;
        }

        .jug-volume-badge {
            background-color: #0f172a;
            border: 1px solid #334155;
            border-radius: 6px;
            padding: 3px 8px;
            font-size: 13px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-main);
        }

        /* Action & Evaluation Display */
        .step-info-card {
            width: 100%;
            max-width: 600px;
            background-color: #111827;
            border: 1px solid #1e293b;
            border-radius: 10px;
            padding: 12px 16px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .step-main-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
        }
        .badge-step-num {
            background-color: var(--accent-blue);
            color: white;
            font-size: 11px;
            font-weight: 800;
            padding: 3px 8px;
            border-radius: 4px;
            text-transform: uppercase;
        }
        .action-desc {
            font-size: 14px;
            font-weight: 700;
            color: var(--text-main);
        }
        .node-eval-row {
            display: flex;
            gap: 12px;
            align-items: center;
            font-size: 12px;
            color: var(--text-muted);
            border-top: 1px solid #1e293b;
            padding-top: 6px;
            font-family: 'JetBrains Mono', monospace;
        }
        .eval-tag {
            color: var(--accent-cyan);
            font-weight: 700;
        }

        /* Playback Controls */
        .controls-toolbar {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            flex-wrap: wrap;
            width: 100%;
        }
        .btn-ctrl {
            background-color: #1e293b;
            border: 1px solid #334155;
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            transition: all 0.15s ease;
        }
        .btn-ctrl:hover {
            background-color: #334155;
            border-color: var(--accent-cyan);
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

        /* Right Panel: Algo Tabs & OPEN / CLOSED Lists */
        .right-panel {
            display: flex;
            flex-direction: column;
            gap: 14px;
        }
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

        /* Solution Steps List */
        .steps-list-box {
            max-height: 200px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 6px;
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
            gap: 6px;
        }
        .oc-title {
            font-size: 12px;
            font-weight: 700;
            color: var(--text-muted);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .oc-list {
            max-height: 110px;
            overflow-y: auto;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: var(--accent-cyan);
            display: flex;
            flex-direction: column;
            gap: 3px;
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
            padding: 12px 16px;
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
            white-space: nowrap;
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
        table.comp-table tbody tr:last-child td {
            border-bottom: none;
        }

        .algo-cell {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            white-space: nowrap;
        }
        .algo-tag {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 48px;
            padding: 2px 8px;
            border-radius: 5px;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        .algo-tag-bfs {
            background-color: rgba(2, 132, 199, 0.18);
            color: #38bdf8;
            border: 1px solid rgba(56, 189, 248, 0.35);
        }
        .algo-tag-ucs {
            background-color: rgba(20, 184, 166, 0.18);
            color: #2dd4bf;
            border: 1px solid rgba(45, 212, 191, 0.35);
        }
        .algo-tag-astar {
            background-color: rgba(168, 85, 247, 0.18);
            color: #c084fc;
            border: 1px solid rgba(192, 132, 252, 0.35);
        }
        .algo-title {
            font-weight: 700;
            color: var(--text-main);
            font-size: 13px;
        }

        .metric-cell {
            display: inline-flex;
            align-items: baseline;
            justify-content: center;
            gap: 4px;
            white-space: nowrap;
        }
        .metric-num {
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            font-weight: 700;
            color: var(--text-main);
        }
        .metric-unit {
            font-size: 11px;
            color: var(--text-muted);
            font-weight: 500;
        }
        .metric-box {
            background-color: #1e293b;
            border: 1px solid #334155;
            padding: 3px 9px;
            border-radius: 6px;
        }

        .badge-optimal {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background-color: rgba(16, 185, 129, 0.12);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.35);
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            white-space: nowrap;
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
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2.2"><path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"/></svg>
                Visualisasi Water Jug (BFS, UCS, A*)
            </h1>
            <p class="subtitle">Eksplorasi Ruang Keadaan dengan Konservasi Volume Air: Jumlah Teko Fleksibel (2 - 5 Teko)</p>
        </header>

        <!-- INPUT CONFIGURATION CARD -->
        <div class="solid-card">
            <div class="config-top">
                <div class="jug-count-selector">
                    <span style="font-size:13px; font-weight:700; color:var(--text-muted); margin-right:4px;">Jumlah Teko:</span>
                    <button class="btn-jug-count" onclick="setJugCount(2)">2 Teko</button>
                    <button class="btn-jug-count active" onclick="setJugCount(3)">3 Teko</button>
                    <button class="btn-jug-count" onclick="setJugCount(4)">4 Teko</button>
                    <button class="btn-jug-count" onclick="setJugCount(5)">5 Teko</button>
                </div>

                <div class="presets-list">
                    <span style="font-size:12px; font-weight:600; color:var(--text-muted);">Preset:</span>
                    <button class="btn-preset" onclick="loadPreset([8, 5, 3], [8, 0, 0], [4, 4, 0])">Classic Poisson (8, 5, 3)</button>
                    <button class="btn-preset" onclick="loadPreset([10, 9, 1], [10, 0, 0], [5, 5, 0])">Kontras BFS vs UCS (10, 9, 1)</button>
                    <button class="btn-preset" onclick="loadPreset([12, 8, 5, 3], [12, 0, 0, 0], [6, 6, 0, 0])">4 Teko (12, 8, 5, 3)</button>
                    <button class="btn-preset" onclick="loadPreset([7, 4], [7, 0], [3, 4])">2 Teko (7, 4)</button>
                    <button class="btn-preset" onclick="loadPreset([8, 5, 3], [8, 0, 0], [5, 5, 0])">Error Total Demo</button>
                </div>
            </div>

            <!-- DYNAMIC JUG INPUTS GRID -->
            <div class="jugs-input-grid" id="jugs-inputs-container">
                <!-- Dynamically populated -->
            </div>

            <!-- WATER CONSERVATION BAR -->
            <div class="water-sum-bar">
                <div style="display:flex; gap:18px; align-items:center;">
                    <span>Total Air Start: <b id="lbl-total-start" class="mono" style="color:var(--accent-cyan);">8 L</b></span>
                    <span>Total Air Goal: <b id="lbl-total-goal" class="mono" style="color:var(--accent-cyan);">8 L</b></span>
                </div>
                <div id="lbl-validation-status" class="status-valid">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
                    Total air seimbang (Aturan Konservasi Terpenuhi)
                </div>
            </div>

            <!-- ERROR BANNER -->
            <div id="error-banner" style="margin-top:12px;">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                <span id="error-text">Pesan kesalahan</span>
            </div>

            <div class="action-row">
                <span style="font-size:12px; color:var(--text-muted); display:inline-flex; align-items:center; gap:6px;">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
                    Hanya ada aksi menuang antar teko (tanpa keran / buang air). Simulasi otomatis diperbarui secara real-time.
                </span>
            </div>
        </div>

        <!-- MAIN SIMULATION STAGE & CONTROLS -->
        <div class="stage-layout">
            <!-- LEFT: JUG WATER ANIMATION & PLAYBACK -->
            <div class="solid-card" style="display:flex; flex-direction:column; gap:16px;">
                <div class="jugs-stage">
                    <!-- JUGS GRAPHICS ROW -->
                    <div class="jugs-row" id="jugs-render-area">
                        <!-- Populated dynamically based on jug count -->
                    </div>

                    <!-- STEP INFO & EVALUATION CARD -->
                    <div class="step-info-card">
                        <div class="step-main-row">
                            <span class="badge-step-num" id="step-badge">LANGKAH 0</span>
                            <span class="action-desc" id="step-action-desc">Kondisi Awal</span>
                        </div>
                        <div class="node-eval-row">
                            <span>State: <b class="eval-tag" id="eval-state">(0, 0, 0)</b></span>
                            <span>g(n): <b class="eval-tag" id="eval-g">0 L</b></span>
                            <span>h(n): <b class="eval-tag" id="eval-h">0</b></span>
                            <span>f(n): <b class="eval-tag" id="eval-f">0</b></span>
                        </div>
                    </div>
                </div>

                <!-- PLAYBACK CONTROLS -->
                <div class="controls-toolbar">
                    <button class="btn-ctrl" onclick="jumpStep(0)">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="19 20 9 12 19 4 19 20"/><line x1="5" y1="19" x2="5" y2="5"/></svg>
                        Awal
                    </button>
                    <button class="btn-ctrl" onclick="prevStep()">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="19 20 9 12 19 4 19 20"/></svg>
                        Prev
                    </button>
                    <button class="btn-ctrl" id="btn-play" onclick="togglePlay()" style="background-color:var(--accent-blue);">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                        Play
                    </button>
                    <button class="btn-ctrl" onclick="nextStep()">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 4 15 12 5 20 5 4"/></svg>
                        Next
                    </button>
                    <button class="btn-ctrl" onclick="jumpEnd()">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 4 15 12 5 20 5 4"/><line x1="19" y1="5" x2="19" y2="19"/></svg>
                        Akhir
                    </button>

                    <div style="display:flex; align-items:center; gap:6px; margin-left:10px;">
                        <span style="font-size:12px; color:var(--text-muted); font-weight:600;">Kecepatan:</span>
                        <select id="sel-speed" onchange="updateSpeed(this.value)" style="background:#1e293b; border:1px solid #334155; color:white; border-radius:6px; padding:4px 8px; font-size:12px;">
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

            <!-- RIGHT: ALGO SELECTOR & OPEN/CLOSED & STEP LIST -->
            <div class="solid-card right-panel">
                <div style="font-size:13px; font-weight:700; color:var(--text-muted); text-transform:uppercase;">
                    Pilih Algoritma Visualisasi:
                </div>
                <div class="algo-tabs">
                    <button class="algo-tab active" id="tab-bfs" onclick="selectAlgoTab('bfs')">BFS</button>
                    <button class="algo-tab" id="tab-ucs" onclick="selectAlgoTab('ucs')">UCS</button>
                    <button class="algo-tab" id="tab-astar" onclick="selectAlgoTab('astar')">A*</button>
                </div>

                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:13px; font-weight:700; color:var(--accent-cyan);">Jalur Solusi Terpilih:</span>
                    <span id="lbl-step-count" style="font-size:12px; color:var(--text-muted); font-weight:600;">0 Langkah</span>
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

        <!-- COMPARATIVE METRICS TABLE -->
        <div class="solid-card">
            <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:14px; flex-wrap:wrap; gap:10px;">
                <h3 style="margin: 0; font-size:16px; color:var(--text-main); display:flex; align-items:center; gap:8px;">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2"><path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/></svg>
                    Tabel Perbandingan Metrik (BFS vs UCS vs A*)
                </h3>
                <span style="font-size:12px; color:var(--text-muted);">
                    Klik baris untuk memuat simulasi algoritma terkait
                </span>
            </div>
            <div class="table-responsive">
                <table class="comp-table">
                    <thead>
                        <tr>
                            <th style="text-align:left;">Algoritma</th>
                            <th style="text-align:center;">Jumlah Langkah</th>
                            <th style="text-align:center;">Total Liter Dipindah (Cost)</th>
                            <th style="text-align:center;">Node Di-expand</th>
                            <th style="text-align:center;">Node Di-generate</th>
                            <th style="text-align:center;">Maks. OPEN</th>
                            <th style="text-align:center;">Waktu Eksekusi</th>
                            <th style="text-align:left;">Hasil Optimal Berdasarkan</th>
                        </tr>
                    </thead>
                    <tbody id="comparison-tbody">
                        <tr><td colspan="8" style="text-align:center; color:var(--text-muted); padding:20px;">Memuat data...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

<script>
    let numJugs = 3;
    let solversData = null;
    let activeAlgo = 'bfs';
    let currentStep = 0;
    let isPlaying = false;
    let timer = null;
    let playSpeed = 900;

    let debounceTimer = null;

    function onInputChange() {
        const isValid = validateAndRecalculate();
        if (isValid) {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => {
                runSolvers();
            }, 300);
        }
    }

    function setJugCount(n) {
        numJugs = n;
        document.querySelectorAll('.btn-jug-count').forEach((b, idx) => {
            b.classList.toggle('active', idx + 2 === n);
        });

        const defaultCaps = [8, 5, 3, 2, 1];
        const defaultStarts = [8, 0, 0, 0, 0];
        const defaultGoals = [4, 4, 0, 0, 0];

        renderInputRows(defaultCaps.slice(0, n), defaultStarts.slice(0, n), defaultGoals.slice(0, n));
        if (validateAndRecalculate()) {
            runSolvers();
        }
    }

    function renderInputRows(caps, starts, goals) {
        const container = document.getElementById('jugs-inputs-container');
        container.innerHTML = '';
        for (let i = 0; i < numJugs; i++) {
            const card = document.createElement('div');
            card.className = 'jug-input-card';
            card.innerHTML = `
                <div class="jug-input-header">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/></svg>
                    Teko ${i + 1}
                </div>
                <div class="row-input">
                    <span>Kapasitas (L):</span>
                    <input type="number" min="1" id="cap-${i}" value="${caps[i] || 5}" oninput="onInputChange()">
                </div>
                <div class="row-input">
                    <span>Isi Start (L):</span>
                    <input type="number" min="0" id="start-${i}" value="${starts[i] || 0}" oninput="onInputChange()">
                </div>
                <div class="row-input">
                    <span>Isi Goal (L):</span>
                    <input type="number" min="0" id="goal-${i}" value="${goals[i] || 0}" oninput="onInputChange()">
                </div>
            `;
            container.appendChild(card);
        }
    }

    function loadPreset(caps, starts, goals) {
        numJugs = caps.length;
        document.querySelectorAll('.btn-jug-count').forEach((b, idx) => {
            b.classList.toggle('active', idx + 2 === numJugs);
        });
        renderInputRows(caps, starts, goals);
        validateAndRecalculate();
        runSolvers();
    }

    function getFormValues() {
        const caps = [], start = [], goal = [];
        for (let i = 0; i < numJugs; i++) {
            caps.push(parseInt(document.getElementById(`cap-${i}`).value) || 0);
            start.push(parseInt(document.getElementById(`start-${i}`).value) || 0);
            goal.push(parseInt(document.getElementById(`goal-${i}`).value) || 0);
        }
        return { caps, start, goal };
    }

    function validateAndRecalculate() {
        const { caps, start, goal } = getFormValues();
        const sumStart = start.reduce((a, b) => a + b, 0);
        const sumGoal = goal.reduce((a, b) => a + b, 0);

        document.getElementById('lbl-total-start').textContent = `${sumStart} L`;
        document.getElementById('lbl-total-goal').textContent = `${sumGoal} L`;

        const statusEl = document.getElementById('lbl-validation-status');
        const errorBanner = document.getElementById('error-banner');
        const submitBtn = document.getElementById('btn-submit');

        let isValid = true;
        let errorMsg = '';

        for (let i = 0; i < numJugs; i++) {
            if (caps[i] <= 0) {
                isValid = false;
                errorMsg = `Kapasitas Teko ${i + 1} harus lebih besar dari 0!`;
                break;
            }
            if (start[i] < 0 || start[i] > caps[i]) {
                isValid = false;
                errorMsg = `Isi start Teko ${i + 1} (${start[i]}L) melebihi kapasitasnya (${caps[i]}L)!`;
                break;
            }
            if (goal[i] < 0 || goal[i] > caps[i]) {
                isValid = false;
                errorMsg = `Isi goal Teko ${i + 1} (${goal[i]}L) melebihi kapasitasnya (${caps[i]}L)!`;
                break;
            }
        }

        if (isValid && sumStart !== sumGoal) {
            isValid = false;
            errorMsg = `Goal ${sumGoal} L, padahal air start ${sumStart} L! Total air harus seimbang.`;
        }

        if (isValid) {
            statusEl.className = 'status-valid';
            statusEl.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
                Total air seimbang (${sumStart} L)
            `;
            errorBanner.style.display = 'none';
            if (submitBtn) submitBtn.disabled = false;
        } else {
            statusEl.className = 'status-invalid';
            statusEl.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                Input Tidak Valid
            `;
            errorBanner.style.display = 'flex';
            document.getElementById('error-text').textContent = errorMsg;
            if (submitBtn) submitBtn.disabled = true;
        }

        return isValid;
    }

    async function runSolvers() {
        stopPlay();
        if (!validateAndRecalculate()) return;

        const { caps, start, goal } = getFormValues();

        try {
            const res = await fetch('/solve', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ caps, start, goal })
            });
            const data = await res.json();

            if (!data.valid) {
                document.getElementById('error-banner').style.display = 'flex';
                document.getElementById('error-text').textContent = data.message;
                return;
            }

            solversData = data;
            renderComparisonTable(data);
            selectAlgoTab(activeAlgo);

        } catch (err) {
            console.error(err);
            document.getElementById('error-banner').style.display = 'flex';
            document.getElementById('error-text').textContent = 'Koneksi ke server terputus: ' + err.message;
        }
    }

    function renderComparisonTable(data) {
        const tbody = document.getElementById('comparison-tbody');
        tbody.innerHTML = '';

        const algos = [
            { key: 'bfs', tag: 'BFS', fullName: 'Breadth-First Search', tagClass: 'algo-tag-bfs', desc: 'Jumlah langkah paling sedikit' },
            { key: 'ucs', tag: 'UCS', fullName: 'Uniform-Cost Search', tagClass: 'algo-tag-ucs', desc: 'Total liter dipindah paling sedikit' },
            { key: 'astar', tag: 'A*', fullName: 'A* Search', tagClass: 'algo-tag-astar', desc: 'Total liter dipindah paling sedikit' }
        ];

        algos.forEach(a => {
            const res = data[a.key];
            const isRowActive = a.key === activeAlgo ? 'class="active-row"' : '';
            if (!res.found) {
                tbody.innerHTML += `
                    <tr ${isRowActive} onclick="selectAlgoTab('${a.key}')">
                        <td>
                            <div class="algo-cell">
                                <span class="algo-tag ${a.tagClass}">${a.tag}</span>
                                <span class="algo-title">${a.fullName}</span>
                            </div>
                        </td>
                        <td colspan="6" style="text-align:center; color:var(--accent-rose); font-weight:700;">Tidak ada solusi</td>
                        <td><span style="font-size:12px; color:var(--text-muted);">${a.desc}</span></td>
                    </tr>
                `;
            } else {
                const timeFormatted = typeof res.time_ms === 'number' ? res.time_ms.toFixed(3) : res.time_ms;
                tbody.innerHTML += `
                    <tr ${isRowActive} onclick="selectAlgoTab('${a.key}')">
                        <td>
                            <div class="algo-cell">
                                <span class="algo-tag ${a.tagClass}">${a.tag}</span>
                                <span class="algo-title">${a.fullName}</span>
                            </div>
                        </td>
                        <td style="text-align:center;">
                            <span class="metric-cell metric-box">
                                <strong class="metric-num">${res.steps}</strong>
                                <span class="metric-unit">langkah</span>
                            </span>
                        </td>
                        <td style="text-align:center;">
                            <span class="metric-cell metric-box">
                                <strong class="metric-num">${res.total_liters}</strong>
                                <span class="metric-unit">L</span>
                            </span>
                        </td>
                        <td style="text-align:center;">
                            <span class="mono metric-num" style="color:#cbd5e1;">${res.expanded}</span>
                        </td>
                        <td style="text-align:center;">
                            <span class="mono metric-num" style="color:#cbd5e1;">${res.generated}</span>
                        </td>
                        <td style="text-align:center;">
                            <span class="mono metric-num" style="color:#cbd5e1;">${res.max_open}</span>
                        </td>
                        <td style="text-align:center;">
                            <span class="metric-cell metric-box">
                                <strong class="metric-num">${timeFormatted}</strong>
                                <span class="metric-unit">ms</span>
                            </span>
                        </td>
                        <td style="text-align:left;">
                            <span class="badge-optimal">
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
                                ${a.desc}
                            </span>
                        </td>
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

        if (!solversData) return;
        renderComparisonTable(solversData);

        const chosen = solversData[algoKey];
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
            const caps = solversData.caps || getFormValues().caps;
            const start = solversData.start || getFormValues().start;
            renderJugsStage(start, caps, null, null);

            const badge = document.getElementById('step-badge');
            badge.textContent = 'TIDAK ADA SOLUSI';
            badge.style.backgroundColor = 'rgba(244, 63, 94, 0.2)';
            badge.style.color = 'var(--accent-rose)';

            document.getElementById('step-action-desc').textContent = 'Goal tidak dapat dicapai dari kondisi awal ini.';
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
                    <span>${step.action}</span>
                </div>
                <span class="mono" style="font-weight:700; color:var(--accent-cyan); font-size:12px;">(${step.state.join(', ')})</span>
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

    function renderJugsStage(state, caps, src, dst) {
        const stageArea = document.getElementById('jugs-render-area');
        stageArea.innerHTML = '';

        for (let i = 0; i < state.length; i++) {
            const vol = state[i];
            const cap = caps[i];
            const pct = Math.min(100, Math.round((vol / cap) * 100));

            const col = document.createElement('div');
            col.className = 'jug-col';

            const isSource = src !== null && src === i;
            const isTarget = dst !== null && dst === i;
            let glassClass = 'jug-glass';
            if (isSource) glassClass += ' source-pour';
            if (isTarget) glassClass += ' target-pour';

            let ticksHtml = '';
            for (let t = 1; t <= cap; t++) {
                const bPct = (t / cap) * 100;
                ticksHtml += `
                    <div class="scale-mark" style="bottom:${bPct}%;"></div>
                    <span class="scale-num" style="bottom:${bPct}%; transform:translateY(50%);">${t}L</span>
                `;
            }

            col.innerHTML = `
                <div class="jug-col-title">
                    Teko ${i + 1} (${cap}L)
                </div>
                <div class="${glassClass}">
                    <div class="measuring-scale">${ticksHtml}</div>
                    <div class="water-column" style="height:${pct}%;">
                        <div class="water-top-line"></div>
                    </div>
                </div>
                <div class="jug-volume-badge">${vol} / ${cap} L</div>
            `;
            stageArea.appendChild(col);
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

        renderJugsStage(step.state, caps, step.src, step.dst);

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

        document.querySelectorAll('.step-row-item').forEach(r => r.classList.remove('active'));
        const activeRow = document.getElementById(`step-row-${currentStep}`);
        if (activeRow) {
            activeRow.classList.add('active');
            activeRow.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }

    function nextStep() {
        if (!solversData || !solversData[activeAlgo]) return;
        const chosen = solversData[activeAlgo];
        if (!chosen.found) return;
        const path = chosen.path;
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

    // Init with default 3-jug Poisson problem
    setJugCount(3);
    runSolvers();
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
    req = request.get_json()
    caps = [int(x) for x in req.get("caps", [8, 5, 3])]
    start = [int(x) for x in req.get("start", [8, 0, 0])]
    goal = [int(x) for x in req.get("goal", [4, 4, 0])]

    is_valid, msg = validate_inputs(caps, start, goal)
    if not is_valid:
        return jsonify({
            "valid": False,
            "message": msg
        })

    bfs_res = solve_bfs(caps, start, goal)
    ucs_res = solve_ucs(caps, start, goal)
    astar_res = solve_astar(caps, start, goal)

    return jsonify({
        "valid": True,
        "message": "Sukses",
        "caps": caps,
        "start": start,
        "goal": goal,
        "bfs": bfs_res,
        "ucs": ucs_res,
        "astar": astar_res
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)