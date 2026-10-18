from flask import Flask, render_template_string, request, jsonify
import math
import time
from collections import deque
import heapq

app = Flask(__name__)

# ==============================================================================
# BAGIAN 1: LOGIKA AI SEARCH - WATER JUG PROBLEM (BFS, UCS, A*)
# ==============================================================================
# Mendukung 2 Mode Target:
# 1. 'single': Mengukur volume tertentu (misal: 4 Liter) di sembarang teko (seperti video short).
# 2. 'exact' : Mencapai state konfigurasi teko tertentu (misal: (4, 4, 0)).
# ==============================================================================

def get_neighbors(state, caps):
    """
    Membangkitkan semua transisi tuang yang valid dari teko i ke teko j.
    Operator tunggal: Menuang dari Teko i ke Teko j (i != j).
    Jumlah air yang dipindah: min(isi_i, kapasitas_j - isi_j).
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
                
                # Narasi deskriptif seperti narator video edukasi
                rem_src = new_state[i]
                curr_dst = new_state[j]
                cap_dst = caps[j]
                status_dst = "hingga penuh" if curr_dst == cap_dst else ""
                status_src = "hingga kosong" if rem_src == 0 else f"(sisa {rem_src}L)"
                
                action_text = f"Tuang {amount}L dari Teko {i+1} ke Teko {j+1} {status_dst} {status_src}".strip()
                neighbors.append((tuple(new_state), amount, action_text, i, j))
    return neighbors

def make_goal_checker(target_mode, target_amount, goal_tuple):
    if target_mode == "single":
        return lambda s: any(v == target_amount for v in s)
    else:
        return lambda s: s == goal_tuple

def make_heuristic(target_mode, target_amount, goal_tuple):
    """
    Heuristik Admisibel A*:
    - Mode 'exact': h(n) = 0.5 * sum(|isi_k(n) - goal_k|)
    - Mode 'single': h(n) = min(|isi_k(n) - target_amount|)
      Karena satu tuangan 'a' liter maksimal mengubah volume teko mana pun sebesar 'a',
      maka selisih ke target paling banyak berkurang 'a' per tuangan. Admisibel!
    """
    if target_mode == "single":
        return lambda s: float(min(abs(v - target_amount) for v in s))
    else:
        return lambda s: sum(abs(curr - g) for curr, g in zip(s, goal_tuple)) / 2.0

def validate_inputs(caps, start, target_mode, target_amount, goal):
    n = len(caps)
    if not (2 <= n <= 5):
        return False, f"Jumlah teko harus antara 2 sampai 5 teko (input: {n} teko)!"
    if len(start) != n:
        return False, "Jumlah nilai start harus sama dengan jumlah teko!"
    
    for i in range(n):
        if caps[i] <= 0:
            return False, f"Kapasitas Teko {i+1} harus lebih besar dari 0 (> 0)!"
        if not (0 <= start[i] <= caps[i]):
            return False, f"Isi awal Teko {i+1} ({start[i]}L) harus antara 0 sampai kapasitasnya ({caps[i]}L)!"

    sum_start = sum(start)
    if sum_start == 0:
        return False, "Total air awal tidak boleh 0 L! Minimal salah satu teko harus berisi air."

    if target_mode == "single":
        if target_amount <= 0:
            return False, "Target volume air yang ingin diukur harus lebih besar dari 0 L!"
        if target_amount > max(caps):
            return False, f"Target {target_amount}L melebihi kapasitas teko terbesar ({max(caps)}L)!"
        if target_amount > sum_start:
            return False, f"Target {target_amount}L melebihi total seluruh air yang tersedia ({sum_start}L)!"
    else:
        if len(goal) != n:
            return False, "Jumlah nilai goal harus sama dengan jumlah teko!"
        for i in range(n):
            if not (0 <= goal[i] <= caps[i]):
                return False, f"Target isi Teko {i+1} ({goal[i]}L) harus antara 0 sampai kapasitas ({caps[i]}L)!"
        sum_goal = sum(goal)
        if sum_start != sum_goal:
            return False, f"Total goal ({sum_goal}L) tidak sama dengan air awal ({sum_start}L)! Menuang air tidak menambah atau mengurangi total air."

    return True, "Valid"

# 1. BREADTH-FIRST SEARCH (BFS)
def solve_bfs(caps, start, is_goal, h_func):
    t0 = time.perf_counter()
    start_t = tuple(start)
    
    if is_goal(start_t):
        return {
            "name": "BFS (Breadth-First Search)",
            "found": True,
            "path": [{"state": list(start_t), "action": "Kondisi Awal (Target Sudah Tercapai)", "amount": 0, "g": 0, "h": 0, "f": 0, "src": None, "dst": None}],
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
    goal_found = None
    
    while queue:
        max_open = max(max_open, len(queue))
        curr = queue.popleft()
        expanded += 1
        
        if is_goal(curr):
            found = True
            goal_found = curr
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
    curr = goal_found
    while curr is not None:
        prev, action, amount, src, dst = came_from[curr]
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
            "dst": dst
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
            "path": [{"state": list(start_t), "action": "Kondisi Awal (Target Sudah Tercapai)", "amount": 0, "g": 0, "h": 0, "f": 0, "src": None, "dst": None}],
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
    goal_found = None
    
    while pq:
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
    curr = goal_found
    while curr is not None:
        prev, action, amount, src, dst = came_from[curr]
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
            "dst": dst
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
            "path": [{"state": list(start_t), "action": "Kondisi Awal (Target Sudah Tercapai)", "amount": 0, "g": 0, "h": 0, "f": 0, "src": None, "dst": None}],
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
    came_from = {start_t: (None, "Mulai (Kondisi Awal)", 0, None, None)}
    g_score = {start_t: 0}
    closed = set()
    
    expanded = 0
    generated = 1
    max_open = 1
    found = False
    goal_found = None
    
    while pq:
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
            
        for nxt, amount, action, src, dst in get_neighbors(curr, caps):
            tentative_g = g_score[curr] + amount
            if nxt not in g_score or tentative_g < g_score[nxt]:
                g_score[nxt] = tentative_g
                h_nxt = h_func(nxt)
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
    curr = goal_found
    while curr is not None:
        prev, action, amount, src, dst = came_from[curr]
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
            "dst": dst
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
        .mode-switcher {
            display: flex;
            background-color: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 12px;
            padding: 4px;
            gap: 4px;
            max-width: 520px;
            width: 100%;
            box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        }
        .mode-btn {
            flex: 1;
            padding: 10px 16px;
            background-color: transparent;
            border: none;
            border-radius: 8px;
            color: var(--text-muted);
            font-size: 14px;
            font-weight: 700;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            transition: all 0.2s ease;
        }
        .mode-btn:hover {
            color: var(--text-main);
            background-color: rgba(255,255,255,0.04);
        }
        .mode-btn.active {
            background-color: var(--accent-blue);
            color: white;
            box-shadow: 0 2px 8px rgba(2, 132, 199, 0.4);
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
            padding: 6px 12px;
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
        .active-template-badge {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            font-size: 11px;
            font-weight: 700;
            padding: 5px 12px;
            border-radius: 20px;
            background: rgba(245, 158, 11, 0.12);
            border: 1px solid rgba(245, 158, 11, 0.35);
            color: #fbbf24;
            margin-left: auto;
            letter-spacing: 0.2px;
            transition: all 0.2s ease;
        }
        .active-template-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background-color: #fbbf24;
            box-shadow: 0 0 8px #fbbf24;
            animation: pulse-dot 2s infinite ease-in-out;
        }
        @keyframes pulse-dot {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.35; transform: scale(0.8); }
        }
        .active-template-badge.custom {
            background: rgba(148, 163, 184, 0.08);
            border-color: rgba(148, 163, 184, 0.2);
            color: #94a3b8;
        }
        .active-template-badge.custom .active-template-dot {
            background-color: #94a3b8;
            box-shadow: none;
            animation: none;
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

        /* Single Target Highlight Box */
        .single-target-box {
            background-color: rgba(56, 189, 248, 0.06);
            border: 1px dashed rgba(56, 189, 248, 0.35);
            border-radius: 10px;
            padding: 12px 18px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 12px;
            margin-bottom: 16px;
        }
        .single-target-title {
            font-size: 14px;
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
                Teka-teki mengukur air menggunakan kombinasi teko (seperti di video Die Hard / Poisson 8-5-3) beserta simulasi dan komparasi algoritma BFS, UCS, dan A*.
            </p>
        </header>

        <!-- MODE SWITCHER (MAIN GAMEPLAY vs AI VISUALIZER) -->
        <div class="mode-switcher-container">
            <div class="mode-switcher">
                <button class="mode-btn active" id="btn-mode-interactive" onclick="switchMainMode('interactive')">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg>
                    Coba Sendiri (Teka-Teki)
                </button>
                <button class="mode-btn" id="btn-mode-ai" onclick="switchMainMode('ai')">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a4 4 0 0 1 4 4v2h2a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2V6a4 4 0 0 1 4-4z"/></svg>
                    Visualisasi AI (BFS / UCS / A*)
                </button>
            </div>
        </div>

        <!-- CONFIGURATION CARD -->
        <div class="solid-card">
            <!-- PRESETS LIST -->
            <div class="presets-bar" id="presets-bar">
                <span style="font-size:12px; font-weight:700; color:var(--text-muted); margin-right:4px;">Preset:</span>
                <button class="preset-chip highlight" id="preset-btn-video" onclick="loadVideoShortsPreset()">
                    Teka-teki Video (8-5-3 Ukur 4L)
                </button>
                <button class="preset-chip" id="preset-btn-poisson" onclick="loadPoissonPreset()">
                    Poisson Klasik (8, 5, 3 &rarr; [4, 4, 0])
                </button>
                <button class="preset-chip" id="preset-btn-contrast" onclick="loadContrastPreset()">
                    Kontras BFS vs UCS (10, 9, 1 &rarr; [5, 5, 0])
                </button>
                <button class="preset-chip" id="preset-btn-4jugs" onclick="load4JugsPreset()">
                    4 Teko (12, 8, 5, 3 &rarr; Ukur 6L)
                </button>
                <button class="preset-chip" id="preset-btn-2jugs" onclick="load2JugsPreset()">
                    2 Teko (7, 4 &rarr; Ukur 3L)
                </button>
                <button class="preset-chip" id="preset-btn-custom" onclick="loadCustomPuzzlePreset()">
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>
                    Custom Puzzle
                </button>

                <!-- Status Template Aktif -->
                <div id="active-template-badge" class="active-template-badge">
                    <span class="active-template-dot"></span>
                    <span id="active-template-text">Template: Teka-teki Video (8-5-3 Ukur 4L)</span>
                </div>
            </div>

            <!-- CUSTOM PUZZLE CONFIGURATION (Hanya muncul jika menu Custom Puzzle diklik) -->
            <div id="custom-puzzle-editor" style="display: none; animation: fadeIn 0.25s ease;">
                <!-- TARGET MODE SELECTOR -->
                <div class="target-mode-row">
                    <div style="display:flex; align-items:center; gap:8px;">
                        <span style="font-size:13px; font-weight:700; color:var(--text-muted);">Jumlah Teko:</span>
                        <button class="preset-chip" id="btn-count-2" onclick="setJugCount(2)">2</button>
                        <button class="preset-chip" id="btn-count-3" onclick="setJugCount(3)">3</button>
                        <button class="preset-chip" id="btn-count-4" onclick="setJugCount(4)">4</button>
                        <button class="preset-chip" id="btn-count-5" onclick="setJugCount(5)">5</button>
                    </div>

                    <div class="target-pills">
                        <button class="target-pill-btn active" id="pill-target-single" onclick="setTargetMode('single')">
                            Target Sederhana (Ukur X Liter)
                        </button>
                        <button class="target-pill-btn" id="pill-target-exact" onclick="setTargetMode('exact')">
                            Target Spesifik Tiap Teko (Exact State)
                        </button>
                    </div>
                </div>

                <!-- TARGET SEDERHANA (SINGLE VOLUME) - PERSIS VIDEO -->
                <div class="single-target-box" id="box-target-single">
                    <div class="single-target-title">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2.2"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>
                        Tujuan Teka-Teki: Ukur volume air di salah satu teko
                    </div>
                    <div class="single-target-control">
                        <span style="font-size:13px; font-weight:600; color:var(--text-muted);">Target Volume:</span>
                        <input type="number" id="input-target-amt" class="input-target-amt" min="1" value="4" oninput="onInputChange()">
                        <span style="font-size:14px; font-weight:700; color:var(--accent-cyan);">Liter</span>
                    </div>
                </div>

                <!-- DYNAMIC JUG INPUTS GRID -->
                <div class="jugs-input-grid" id="jugs-inputs-container">
                    <!-- Dynamically populated -->
                </div>

                <!-- WATER CONSERVATION & VALIDATION STATUS -->
                <div class="water-sum-bar">
                    <div style="display:flex; gap:16px; align-items:center;">
                        <span>Total Air Awal: <b id="lbl-total-start" class="mono" style="color:var(--accent-cyan);">8 L</b></span>
                        <span id="wrap-total-goal" style="display:none;">Total Air Goal: <b id="lbl-total-goal" class="mono" style="color:var(--accent-cyan);">8 L</b></span>
                    </div>
                    <div id="lbl-validation-status" class="status-valid">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
                        Siap Disimulasikan
                    </div>
                </div>

                <!-- ERROR BANNER -->
                <div id="error-banner">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                    <span id="error-text">Pesan kesalahan</span>
                </div>
            </div>
        </div>

        <!-- MAIN INTERACTIVE STAGE & VISUALIZATION -->
        <div class="stage-card">
            <!-- STAGE HUD -->
            <div class="stage-hud">
                <div class="hud-status" id="hud-status-text">
                    <span style="color:var(--accent-cyan); font-weight:800;">Target:</span>
                    <span id="hud-target-desc">Ukur tepat 4 Liter pada salah satu teko</span>
                </div>
                <div class="hud-stats">
                    <div id="hud-interactive-stats" style="display:flex; gap:12px;">
                        <span>Langkah: <b class="stat-badge" id="player-move-count">0</b></span>
                        <span>Air Dipindah: <b class="stat-badge" id="player-liter-count">0 L</b></span>
                    </div>
                    <div id="hud-ai-stats" style="display:none;">
                        <span>Algoritma Aktif: <b class="stat-badge" id="hud-active-algo-name">BFS</b></span>
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

            <!-- MODE 1: INTERACTIVE PUZZLE CONTROLS ("COBA SENDIRI") -->
            <div class="interactive-controls-bar" id="interactive-controls-section">
                <button class="btn-action" onclick="undoPlayerMove()" id="btn-undo" disabled>
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M3 7v6h6"/><path d="M21 17a9 9 0 0 0-9-9 9 9 0 0 0-6 2.3L3 13"/></svg>
                    Batalkan (Undo)
                </button>
                <button class="btn-action" onclick="resetPlayerGame()">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M1 4v6h6"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/></svg>
                    Ulang dari Awal (Reset)
                </button>
                <button class="btn-action" onclick="givePlayerHint()" id="btn-hint">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><line x1="9" y1="18" x2="15" y2="18"/><line x1="10" y1="22" x2="14" y2="22"/><path d="M15.09 14c.18-.98.65-1.74 1.41-2.5A4.65 4.65 0 0 0 18 8 6 6 0 0 0 6 8c0 1 .23 2.23 1.5 3.5A4.61 4.61 0 0 1 8.91 14"/></svg>
                    Beri Petunjuk (Hint)
                </button>
                <button class="btn-action primary" onclick="switchMainMode('ai')">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                    Lihat Solusi Lengkap AI
                </button>
            </div>

            <!-- MODE 2: AI STEP NARRATION & PLAYBACK CONTROLS -->
            <div id="ai-controls-section" style="display:none; flex-direction:column; gap:12px;">
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
        <div id="ai-inspect-section" class="ai-inspect-layout" style="display:none;">
            <!-- LEFT: COMPARISON TABLE -->
            <div class="solid-card" style="display:flex; flex-direction:column; gap:12px;">
                <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:8px;">
                    <h3 style="margin: 0; font-size:15px; color:var(--text-main); display:flex; align-items:center; gap:8px;">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2"><path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/></svg>
                        Tabel Perbandingan Metrik (BFS vs UCS vs A*)
                    </h3>
                    <span style="font-size:12px; color:var(--text-muted);">
                        Klik baris tabel untuk meninjau jalur algoritma
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

    <!-- VICTORY MODAL -->
    <div class="modal-backdrop" id="victory-modal">
        <div class="modal-card">
            <div style="font-size:48px;">🏆</div>
            <h2 style="margin:0; font-size:22px; color:var(--accent-green);">Teka-Teki Berhasil Dipecahkan!</h2>
            <p id="victory-msg" style="margin:0; font-size:14px; color:var(--text-muted); line-height:1.5;">
                Anda berhasil mengukur volume air yang diinginkan!
            </p>
            <div style="display:flex; gap:10px; width:100%; justify-content:center; margin-top:8px;">
                <button class="btn-action" onclick="closeVictoryModal(); resetPlayerGame();">
                    Main Lagi
                </button>
                <button class="btn-action primary" onclick="closeVictoryModal(); switchMainMode('ai');">
                    Tonton Solusi AI
                </button>
            </div>
        </div>
    </div>

<script>
    // State Aplikasi
    let currentMainMode = 'interactive'; // 'interactive' atau 'ai'
    let targetMode = 'single'; // 'single' atau 'exact'
    let numJugs = 3;
    let solversData = null;

    // State Pemain (Mode Interaktif)
    let playerInitialState = [8, 0, 0];
    let playerCurrentState = [8, 0, 0];
    let playerHistory = [];
    let playerTotalLiters = 0;
    let selectedSourceJug = null; // indeks teko sumber yang dipilih (0..n-1)

    // State AI Playback
    let activeAlgo = 'bfs';
    let currentStep = 0;
    let isPlaying = false;
    let timer = null;
    let playSpeed = 900;
    let debounceTimer = null;

    // --- DEFINISI PRESET TEMPLATE & AKTIVASI INSTAN ---
    const PRESETS = {
        'video': {
            name: 'Teka-teki Video (8-5-3 Ukur 4L)',
            targetMode: 'single',
            numJugs: 3,
            targetAmt: 4,
            caps: [8, 5, 3],
            start: [8, 0, 0],
            goal: [4, 4, 0]
        },
        'poisson': {
            name: 'Poisson Klasik (8, 5, 3 → [4, 4, 0])',
            targetMode: 'exact',
            numJugs: 3,
            targetAmt: 4,
            caps: [8, 5, 3],
            start: [8, 0, 0],
            goal: [4, 4, 0]
        },
        'contrast': {
            name: 'Kontras BFS vs UCS (10, 9, 1 → [5, 5, 0])',
            targetMode: 'exact',
            numJugs: 3,
            targetAmt: 5,
            caps: [10, 9, 1],
            start: [10, 0, 0],
            goal: [5, 5, 0]
        },
        '4jugs': {
            name: '4 Teko (12, 8, 5, 3 → Ukur 6L)',
            targetMode: 'single',
            numJugs: 4,
            targetAmt: 6,
            caps: [12, 8, 5, 3],
            start: [12, 0, 0, 0],
            goal: [6, 6, 0, 0]
        },
        '2jugs': {
            name: '2 Teko (7, 4 → Ukur 3L)',
            targetMode: 'single',
            numJugs: 2,
            targetAmt: 3,
            caps: [7, 4],
            start: [7, 0],
            goal: [3, 4]
        }
    };

    let activePresetKey = 'video';
    let solverAbortController = null;

    function setActivePresetUI(presetKey) {
        activePresetKey = presetKey;
        const keys = ['video', 'poisson', 'contrast', '4jugs', '2jugs', 'custom'];
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

        // Tampilkan konfigurasi hanya jika mode Custom Puzzle dipilih, jika preset maka disembunyikan
        const editor = document.getElementById('custom-puzzle-editor');
        if (editor) {
            editor.style.display = (presetKey === 'custom') ? 'block' : 'none';
        }

        const presetsBar = document.getElementById('presets-bar');
        if (presetsBar) {
            presetsBar.classList.toggle('with-border', presetKey === 'custom');
        }

        const badge = document.getElementById('active-template-badge');
        const textEl = document.getElementById('active-template-text');
        if (badge && textEl) {
            if (presetKey === 'custom') {
                badge.className = 'active-template-badge custom';
                textEl.textContent = 'Mode: Custom Puzzle';
            } else if (PRESETS[presetKey]) {
                badge.className = 'active-template-badge';
                textEl.textContent = `Template: ${PRESETS[presetKey].name}`;
            }
        }
    }

    function loadCustomPuzzlePreset() {
        setActivePresetUI('custom');
        const container = document.getElementById('jugs-inputs-container');
        if (!container || container.children.length === 0) {
            const { caps, start, goal } = getFormValues();
            renderInputRows(caps, start, goal);
        }
    }

    function applyPreset(key) {
        const p = PRESETS[key];
        if (!p) return;

        // 1. Langsung ubah highlight & badge seketika (0ms delay) dan sembunyikan menu kustom
        setActivePresetUI(key);

        // 2. Hentikan pemutaran animasi / aksi yang sedang berjalan
        stopPlay();
        selectedSourceJug = null;

        // 3. Set konfigurasi state
        targetMode = p.targetMode;
        numJugs = p.numJugs;

        updateTargetModeUI();
        updateJugCountUI();

        const amtInput = document.getElementById('input-target-amt');
        if (amtInput) amtInput.value = p.targetAmt;

        renderInputRows(p.caps, p.start, p.goal);

        // 4. Update visual langsung tanpa delay (0ms)
        playerInitialState = [...p.start];
        playerCurrentState = [...p.start];
        playerHistory = [];
        playerTotalLiters = 0;
        updatePlayerHUD();

        if (currentMainMode === 'interactive') {
            renderInteractiveJugs(playerCurrentState, p.caps);
        } else {
            currentStep = 0;
            renderAIJugsStage(p.start, p.caps, null, null);
            document.getElementById('lbl-curr-step').textContent = '0';
            const scrubber = document.getElementById('scrubber');
            if (scrubber) scrubber.value = 0;
        }

        // 5. Jalankan backend solver langsung tanpa jeda debounce 250ms
        triggerSolveAndSync(true);
    }

    function loadVideoShortsPreset() { applyPreset('video'); }
    function loadPoissonPreset() { applyPreset('poisson'); }
    function loadContrastPreset() { applyPreset('contrast'); }
    function load4JugsPreset() { applyPreset('4jugs'); }
    function load2JugsPreset() { applyPreset('2jugs'); }

    function setJugCount(n) {
        setActivePresetUI('custom');
        stopPlay();
        selectedSourceJug = null;
        numJugs = n;
        updateJugCountUI();
        const defaultCaps = [8, 5, 3, 2, 1];
        const defaultStarts = [8, 0, 0, 0, 0];
        const defaultGoals = [4, 4, 0, 0, 0];
        const caps = defaultCaps.slice(0, n);
        const starts = defaultStarts.slice(0, n);
        const goals = defaultGoals.slice(0, n);
        renderInputRows(caps, starts, goals);

        playerInitialState = [...starts];
        playerCurrentState = [...starts];
        playerHistory = [];
        playerTotalLiters = 0;
        updatePlayerHUD();
        if (currentMainMode === 'interactive') {
            renderInteractiveJugs(playerCurrentState, caps);
        } else {
            currentStep = 0;
            renderAIJugsStage(starts, caps, null, null);
            document.getElementById('lbl-curr-step').textContent = '0';
            const scrubber = document.getElementById('scrubber');
            if (scrubber) scrubber.value = 0;
        }
        triggerSolveAndSync(true);
    }

    function updateJugCountUI() {
        for (let i = 2; i <= 5; i++) {
            const btn = document.getElementById(`btn-count-${i}`);
            if (btn) {
                btn.style.backgroundColor = (i === numJugs) ? 'var(--accent-blue)' : '#1e293b';
                btn.style.color = (i === numJugs) ? 'white' : 'var(--text-main)';
            }
        }
    }

    function setTargetMode(mode) {
        setActivePresetUI('custom');
        targetMode = mode;
        updateTargetModeUI();
        triggerSolveAndSync(true);
    }

    function updateTargetModeUI() {
        document.getElementById('pill-target-single').classList.toggle('active', targetMode === 'single');
        document.getElementById('pill-target-exact').classList.toggle('active', targetMode === 'exact');
        document.getElementById('box-target-single').style.display = (targetMode === 'single') ? 'flex' : 'none';
        document.getElementById('wrap-total-goal').style.display = (targetMode === 'exact') ? 'inline' : 'none';

        // Tampilkan/sembunyikan input goal di kartu teko
        document.querySelectorAll('.row-goal-input').forEach(el => {
            el.style.display = (targetMode === 'exact') ? 'flex' : 'none';
        });

        // Update teks HUD
        if (targetMode === 'single') {
            const amt = document.getElementById('input-target-amt').value || 4;
            document.getElementById('hud-target-desc').textContent = `Ukur tepat ${amt} Liter pada salah satu teko`;
        } else {
            document.getElementById('hud-target-desc').textContent = `Capai kondisi akhir teko yang ditentukan`;
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
                    <span>Isi Awal (L):</span>
                    <input type="number" min="0" id="start-${i}" value="${starts[i] || 0}" oninput="onInputChange()">
                </div>
                <div class="row-input row-goal-input" style="${targetMode === 'exact' ? 'display:flex;' : 'display:none;'}">
                    <span>Target Goal (L):</span>
                    <input type="number" min="0" id="goal-${i}" value="${goals[i] || 0}" oninput="onInputChange()">
                </div>
            `;
            container.appendChild(card);
        }
    }

    function onInputChange() {
        setActivePresetUI('custom');
        if (targetMode === 'single') {
            const amt = document.getElementById('input-target-amt').value || 4;
            document.getElementById('hud-target-desc').textContent = `Ukur tepat ${amt} Liter pada salah satu teko`;
        }
        triggerSolveAndSync(false);
    }

    function triggerSolveAndSync(immediate = false) {
        clearTimeout(debounceTimer);
        if (immediate) {
            runSolvers();
        } else {
            debounceTimer = setTimeout(() => {
                runSolvers();
            }, 250);
        }
    }

    function getFormValues() {
        const caps = [], start = [], goal = [];
        for (let i = 0; i < numJugs; i++) {
            caps.push(parseInt(document.getElementById(`cap-${i}`).value) || 0);
            start.push(parseInt(document.getElementById(`start-${i}`).value) || 0);
            goal.push(parseInt(document.getElementById(`goal-${i}`).value) || 0);
        }
        const targetAmount = parseInt(document.getElementById('input-target-amt').value) || 4;
        return { caps, start, goal, targetAmount };
    }

    function validateFrontendInputs() {
        const { caps, start, goal, targetAmount } = getFormValues();
        const sumStart = start.reduce((a, b) => a + b, 0);
        const sumGoal = goal.reduce((a, b) => a + b, 0);

        document.getElementById('lbl-total-start').textContent = `${sumStart} L`;
        document.getElementById('lbl-total-goal').textContent = `${sumGoal} L`;

        const statusEl = document.getElementById('lbl-validation-status');
        const errorBanner = document.getElementById('error-banner');

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
                errorMsg = `Isi awal Teko ${i + 1} (${start[i]}L) melebihi kapasitasnya (${caps[i]}L)!`;
                break;
            }
        }

        if (isValid && sumStart === 0) {
            isValid = false;
            errorMsg = 'Minimal harus ada salah satu teko yang berisi air!';
        }

        if (isValid) {
            if (targetMode === 'single') {
                if (targetAmount <= 0) {
                    isValid = false;
                    errorMsg = 'Target volume yang ingin diukur harus lebih dari 0 L!';
                } else if (targetAmount > Math.max(...caps)) {
                    isValid = false;
                    errorMsg = `Target ${targetAmount}L melebihi kapasitas teko terbesar (${Math.max(...caps)}L)!`;
                } else if (targetAmount > sumStart) {
                    isValid = false;
                    errorMsg = `Target ${targetAmount}L melebihi total air yang tersedia (${sumStart}L)!`;
                }
            } else {
                for (let i = 0; i < numJugs; i++) {
                    if (goal[i] < 0 || goal[i] > caps[i]) {
                        isValid = false;
                        errorMsg = `Isi target Teko ${i + 1} (${goal[i]}L) melebihi kapasitasnya (${caps[i]}L)!`;
                        break;
                    }
                }
                if (isValid && sumStart !== sumGoal) {
                    isValid = false;
                    errorMsg = `Total air goal (${sumGoal}L) tidak sama dengan air awal (${sumStart}L)! Menuang air tidak menambah/mengurangi volume total.`;
                }
            }
        }

        if (isValid) {
            statusEl.className = 'status-valid';
            statusEl.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
                Aturan Air Terpenuhi (${sumStart} L)
            `;
            errorBanner.style.display = 'none';
        } else {
            statusEl.className = 'status-invalid';
            statusEl.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                Input Belum Tepat
            `;
            errorBanner.style.display = 'flex';
            document.getElementById('error-text').textContent = errorMsg;
        }

        return isValid;
    }

    // --- CALL BACKEND SOLVER ---
    async function runSolvers() {
        if (!validateFrontendInputs()) return;
        const { caps, start, goal, targetAmount } = getFormValues();

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
                    caps,
                    start,
                    target_mode: targetMode,
                    target_amount: targetAmount,
                    goal
                })
            });
            const data = await res.json();

            if (!data.valid) {
                document.getElementById('error-banner').style.display = 'flex';
                document.getElementById('error-text').textContent = data.message;
                return;
            }

            solversData = data;
            renderComparisonTable(data);

            if (currentMainMode === 'ai') {
                selectAlgoTab(activeAlgo);
            } else if (currentMainMode === 'interactive') {
                const activeCaps = data.caps || caps;
                renderInteractiveJugs(playerCurrentState, activeCaps);
            }

        } catch (err) {
            if (err.name !== 'AbortError') {
                console.error(err);
            }
        }
    }

    // --- MODE SWITCHER (INTERACTIVE VS AI) ---
    function switchMainMode(mode) {
        currentMainMode = mode;
        stopPlay();
        selectedSourceJug = null;

        document.getElementById('btn-mode-interactive').classList.toggle('active', mode === 'interactive');
        document.getElementById('btn-mode-ai').classList.toggle('active', mode === 'ai');

        document.getElementById('interactive-controls-section').style.display = (mode === 'interactive') ? 'flex' : 'none';
        document.getElementById('ai-controls-section').style.display = (mode === 'ai') ? 'flex' : 'none';
        document.getElementById('ai-inspect-section').style.display = (mode === 'ai') ? 'grid' : 'none';

        document.getElementById('hud-interactive-stats').style.display = (mode === 'interactive') ? 'flex' : 'none';
        document.getElementById('hud-ai-stats').style.display = (mode === 'ai') ? 'flex' : 'none';

        if (mode === 'interactive') {
            const caps = solversData ? solversData.caps : getFormValues().caps;
            renderInteractiveJugs(playerCurrentState, caps);
        } else {
            selectAlgoTab(activeAlgo);
        }
    }

    // ==============================================================================
    // LOGIKA MODE 1: INTERAKTIF (COBA SENDIRI SEPERTI DI VIDEO SHORTS)
    // ==============================================================================
    function resetPlayerGame() {
        const { start } = getFormValues();
        playerInitialState = [...start];
        playerCurrentState = [...start];
        playerHistory = [];
        playerTotalLiters = 0;
        selectedSourceJug = null;

        updatePlayerHUD();
        const caps = solversData ? solversData.caps : getFormValues().caps;
        renderInteractiveJugs(playerCurrentState, caps);
    }

    function updatePlayerHUD() {
        document.getElementById('player-move-count').textContent = playerHistory.length;
        document.getElementById('player-liter-count').textContent = `${playerTotalLiters} L`;
        document.getElementById('btn-undo').disabled = (playerHistory.length === 0);
    }

    function onJugClick(jugIndex) {
        if (currentMainMode !== 'interactive') return;
        const { caps } = getFormValues();

        // 1. Jika belum ada teko sumber yang dipilih:
        if (selectedSourceJug === null) {
            if (playerCurrentState[jugIndex] === 0) {
                showPourIndicator(`Teko ${jugIndex + 1} kosong! Pilih teko yang memiliki air.`);
                return;
            }
            selectedSourceJug = jugIndex;
            showPourIndicator(`Teko ${jugIndex + 1} dipilih. Sekarang klik teko tujuan untuk menuang air.`);
            renderInteractiveJugs(playerCurrentState, caps);
            return;
        }

        // 2. Jika mengklik teko yang sama -> batalkan pilihan
        if (selectedSourceJug === jugIndex) {
            selectedSourceJug = null;
            hidePourIndicator();
            renderInteractiveJugs(playerCurrentState, caps);
            return;
        }

        // 3. Eksekusi tuang dari teko sumber ke teko tujuan
        const src = selectedSourceJug;
        const dst = jugIndex;

        if (playerCurrentState[dst] >= caps[dst]) {
            showPourIndicator(`Teko ${dst + 1} sudah penuh (${caps[dst]}L)! Pilih teko lain.`);
            return;
        }

        const amount = Math.min(playerCurrentState[src], caps[dst] - playerCurrentState[dst]);
        if (amount <= 0) {
            showPourIndicator(`Tidak bisa menuang air ke Teko ${dst + 1}.`);
            return;
        }

        // Simpan riwayat untuk Undo
        playerHistory.push({
            state: [...playerCurrentState],
            poured: amount,
            src: src,
            dst: dst
        });

        playerCurrentState[src] -= amount;
        playerCurrentState[dst] += amount;
        playerTotalLiters += amount;
        selectedSourceJug = null;

        updatePlayerHUD();
        showPourIndicator(`Menuang ${amount}L dari Teko ${src + 1} ke Teko ${dst + 1}`);
        renderInteractiveJugs(playerCurrentState, caps);

        // Cek apakah target teka-teki tercapai!
        checkPlayerVictory();
    }

    function undoPlayerMove() {
        if (playerHistory.length === 0) return;
        const last = playerHistory.pop();
        playerCurrentState = [...last.state];
        playerTotalLiters = Math.max(0, playerTotalLiters - last.poured);
        selectedSourceJug = null;

        updatePlayerHUD();
        showPourIndicator('Langkah terakhir dibatalkan.');
        const caps = solversData ? solversData.caps : getFormValues().caps;
        renderInteractiveJugs(playerCurrentState, caps);
    }

    function givePlayerHint() {
        if (!solversData || !solversData.bfs || !solversData.bfs.found) {
            alert('Tidak ditemukan jalur solusi untuk konfigurasi ini.');
            return;
        }

        const path = solversData.bfs.path;
        // Cari langkah optimal berikutnya dari current state jika ada di jalur solusi
        const currStr = playerCurrentState.join(',');
        let matchIdx = -1;
        for (let i = 0; i < path.length; i++) {
            if (path[i].state.join(',') === currStr) {
                matchIdx = i;
                break;
            }
        }

        if (matchIdx !== -1 && matchIdx < path.length - 1) {
            const nextMove = path[matchIdx + 1];
            showPourIndicator(`💡 Petunjuk: ${nextMove.action}`);
            selectedSourceJug = nextMove.src;
            const caps = solversData.caps;
            renderInteractiveJugs(playerCurrentState, caps);
        } else {
            showPourIndicator(`💡 Petunjuk: Kembali ke awal atau ikuti solusi AI (${solversData.bfs.steps} langkah optimal).`);
        }
    }

    function checkPlayerVictory() {
        const { targetAmount, goal } = getFormValues();
        let isWin = false;

        if (targetMode === 'single') {
            isWin = playerCurrentState.some(v => v === targetAmount);
        } else {
            isWin = playerCurrentState.every((v, i) => v === goal[i]);
        }

        if (isWin) {
            setTimeout(() => {
                const optimalSteps = (solversData && solversData.bfs && solversData.bfs.found) ? solversData.bfs.steps : '-';
                document.getElementById('victory-msg').innerHTML = `
                    Anda berhasil menyelesaikan teka-teki dalam <b>${playerHistory.length} langkah</b> (${playerTotalLiters} L air dipindah)!<br>
                    <span style="font-size:12px; color:var(--accent-cyan);">Solusi optimal AI: ${optimalSteps} langkah.</span>
                `;
                document.getElementById('victory-modal').style.display = 'flex';
            }, 300);
        }
    }

    function closeVictoryModal() {
        document.getElementById('victory-modal').style.display = 'none';
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
        // Setiap 2 L kapasitas menambah tinggi teko sebesar 30px dan lebar 6px
        // Base: 90px height, 76px width
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
                // Teks HANYA ditampilkan ketika kelipatan 2 saja
                // Pastikan teks tidak terpotong jika dekat bibir atas (misal t === cap)
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

    function renderInteractiveJugs(state, caps) {
        const stageArea = document.getElementById('jugs-render-area');
        stageArea.innerHTML = '';
        const { targetAmount, goal } = getFormValues();

        for (let i = 0; i < state.length; i++) {
            const vol = state[i];
            const cap = caps[i];
            const pct = Math.min(100, Math.round((vol / cap) * 100));
            const dim = getJugDimensions(cap);
            const ticksHtml = getMeasuringScaleHtml(cap);

            const isSource = (selectedSourceJug === i);
            const isPotentialTarget = (selectedSourceJug !== null && selectedSourceJug !== i && vol < cap);
            
            let isTargetAchieved = false;
            if (targetMode === 'single') {
                isTargetAchieved = (vol === targetAmount);
            } else {
                isTargetAchieved = (vol === goal[i]);
            }

            const item = document.createElement('div');
            item.className = 'jug-item';
            if (isSource) item.classList.add('selected-source');
            if (isPotentialTarget) item.classList.add('potential-target');
            if (isTargetAchieved) item.classList.add('target-reached');

            item.onclick = () => onJugClick(i);

            item.innerHTML = `
                <div class="jug-tag-badge">Sumber Tuang</div>
                <div class="jug-win-badge">Target (${vol}L)</div>
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


    // ==============================================================================
    // LOGIKA MODE 2: SIMULASI AI SEARCH (BFS, UCS, A*)
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
            if (!res.found) {
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
            renderAIJugsStage(start, caps, null, null);

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

    function renderAIJugsStage(state, caps, src, dst) {
        const stageArea = document.getElementById('jugs-render-area');
        stageArea.innerHTML = '';
        const { targetAmount, goal } = getFormValues();

        for (let i = 0; i < state.length; i++) {
            const vol = state[i];
            const cap = caps[i];
            const pct = Math.min(100, Math.round((vol / cap) * 100));
            const dim = getJugDimensions(cap);
            const ticksHtml = getMeasuringScaleHtml(cap);

            const isSource = (src !== null && src === i);
            const isTarget = (dst !== null && dst === i);

            let isTargetAchieved = false;
            if (targetMode === 'single') {
                isTargetAchieved = (vol === targetAmount);
            } else {
                isTargetAchieved = (vol === goal[i]);
            }

            const item = document.createElement('div');
            item.className = 'jug-item';
            if (isSource) item.classList.add('ai-source');
            if (isTarget) item.classList.add('ai-target');
            if (isTargetAchieved) item.classList.add('target-reached');

            item.innerHTML = `
                ${isTargetAchieved ? `<div class="jug-win-badge">Target (${vol}L)</div>` : ''}
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

        renderAIJugsStage(step.state, caps, step.src, step.dst);

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

    // Inisialisasi awal dengan Preset Video Shorts (8, 5, 3 &rarr; Ukur 4L)
    loadVideoShortsPreset();
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
    start = [int(x) for x in req.get("start", [8, 0, 0])]
    target_mode = req.get("target_mode", "single")
    target_amount = int(req.get("target_amount", 4))
    goal = [int(x) for x in req.get("goal", [4, 4, 0])]

    is_valid, msg = validate_inputs(caps, start, target_mode, target_amount, goal)
    if not is_valid:
        return jsonify({
            "valid": False,
            "message": msg
        })

    goal_checker = make_goal_checker(target_mode, target_amount, tuple(goal))
    heuristic_func = make_heuristic(target_mode, target_amount, tuple(goal))

    bfs_res = solve_bfs(caps, start, goal_checker, heuristic_func)
    ucs_res = solve_ucs(caps, start, goal_checker, heuristic_func)
    astar_res = solve_astar(caps, start, goal_checker, heuristic_func)

    return jsonify({
        "valid": True,
        "message": "Sukses",
        "caps": caps,
        "start": start,
        "target_mode": target_mode,
        "target_amount": target_amount,
        "goal": goal,
        "bfs": bfs_res,
        "ucs": ucs_res,
        "astar": astar_res
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)