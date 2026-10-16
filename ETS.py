from flask import Flask, render_template_string, request, jsonify
import heapq
import time
from collections import deque

app = Flask(__name__)

# ==========================================
# BAGIAN 1: LOGIKA AI SEARCH (MURNI PYTHON)
# ==========================================
# Keterangan Grid:
# 0 = Server Normal (Cost: 1)
# 1 = Firewall / Tembok (Tidak bisa dilewati)
# 2 = IDS / Honeypot / Antivirus (Cost: 10 -> Risiko Terdeteksi Tinggi!)

COST_MAP = {
    0: 1,   # Jalur aman
    2: 10   # Jalur berisiko tinggi (IDS/Antivirus)
}

def get_neighbors(r, c, rows, cols, grid):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neighbors = []
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != 1:
            neighbors.append((nr, nc))
    return neighbors

def reconstruct_path(came_from, start, goal):
    curr = goal
    path = []
    while curr is not None:
        path.append(list(curr))
        curr = came_from.get(curr)
    path.reverse()
    return path

def calculate_path_cost(path, grid, start):
    if not path:
        return 0
    total = 0
    for r, c in path:
        if (r, c) != start:
            total += COST_MAP.get(grid[r][c], 1)
    return total

# 1. Breadth-First Search (Uninformed - Abaikan Cost Risiko)
def solve_bfs(grid, start, goal):
    t0 = time.perf_counter()
    rows, cols = len(grid), len(grid[0])
    queue = deque([start])
    came_from = {start: None}
    visited_order = []

    while queue:
        curr = queue.popleft()
        visited_order.append(list(curr))

        if curr == goal:
            break

        for nxt in get_neighbors(curr[0], curr[1], rows, cols, grid):
            if nxt not in came_from:
                came_from[nxt] = curr
                queue.append(nxt)

    exec_time = (time.perf_counter() - t0) * 1000
    path = reconstruct_path(came_from, start, goal) if goal in came_from else []
    total_cost = calculate_path_cost(path, grid, start)
    return {
        "name": "BFS (Breadth-First Search)",
        "path": path,
        "visited": visited_order,
        "cost": total_cost,
        "steps": max(0, len(path) - 1),
        "expanded": len(visited_order),
        "time_ms": round(exec_time, 3)
    }

# 2. Uniform-Cost Search (Uninformed - Cari Risiko Terendah)
def solve_ucs(grid, start, goal):
    t0 = time.perf_counter()
    rows, cols = len(grid), len(grid[0])
    pq = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    visited_order = []
    closed_set = set()

    while pq:
        curr_cost, curr = heapq.heappop(pq)
        if curr in closed_set:
            continue
        closed_set.add(curr)
        visited_order.append(list(curr))

        if curr == goal:
            break

        for nxt in get_neighbors(curr[0], curr[1], rows, cols, grid):
            new_cost = cost_so_far[curr] + COST_MAP.get(grid[nxt[0]][nxt[1]], 1)
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                came_from[nxt] = curr
                heapq.heappush(pq, (new_cost, nxt))

    exec_time = (time.perf_counter() - t0) * 1000
    path = reconstruct_path(came_from, start, goal) if goal in came_from else []
    total_cost = cost_so_far.get(goal, 0)
    return {
        "name": "UCS (Uniform-Cost Search)",
        "path": path,
        "visited": visited_order,
        "cost": total_cost,
        "steps": max(0, len(path) - 1),
        "expanded": len(visited_order),
        "time_ms": round(exec_time, 3)
    }

# 3. A* Search (Informed - Risiko Terendah + Manhattan Heuristic)
def solve_astar(grid, start, goal):
    t0 = time.perf_counter()
    rows, cols = len(grid), len(grid[0])

    def heuristic(a, b):
        # Manhattan Distance (Admissible karena cost minimal per langkah = 1)
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    pq = [(heuristic(start, goal), 0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    visited_order = []
    closed_set = set()

    while pq:
        _, curr_cost, curr = heapq.heappop(pq)
        if curr in closed_set:
            continue
        closed_set.add(curr)
        visited_order.append(list(curr))

        if curr == goal:
            break

        for nxt in get_neighbors(curr[0], curr[1], rows, cols, grid):
            new_cost = cost_so_far[curr] + COST_MAP.get(grid[nxt[0]][nxt[1]], 1)
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + heuristic(nxt, goal)
                came_from[nxt] = curr
                heapq.heappush(pq, (priority, new_cost, nxt))

    exec_time = (time.perf_counter() - t0) * 1000
    path = reconstruct_path(came_from, start, goal) if goal in came_from else []
    total_cost = cost_so_far.get(goal, 0)
    return {
        "name": "A* Search (Manhattan Heuristic)",
        "path": path,
        "visited": visited_order,
        "cost": total_cost,
        "steps": max(0, len(path) - 1),
        "expanded": len(visited_order),
        "time_ms": round(exec_time, 3)
    }


# ==========================================
# BAGIAN 2: TEMPLATE HTML + CSS + JAVASCRIPT
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <title>Stealth Hacker Grid - AI Search Visualizer</title>
    <style>
        :root {
            --bg-dark: #0f172a;
            --panel-bg: #1e293b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #38bdf8;
            --hacker-green: #22c55e;
            --goal-gold: #eab308;
            --firewall: #334155;
            --ids-trap: #f43f5e;
            --visited: rgba(56, 189, 248, 0.35);
            --path: #4ade80;
        }
        * { box-sizing: border-box; font-family: 'Segoe UI', system-ui, sans-serif; }
        body {
            margin: 0; padding: 24px;
            background-color: var(--bg-dark); color: var(--text-main);
            display: flex; flex-direction: column; align-items: center;
        }
        h1 { margin: 0 0 6px 0; font-size: 24px; color: var(--accent); display: flex; align-items: center; gap: 8px; }
        p.sub { margin: 0 0 20px 0; color: var(--text-muted); font-size: 14px; }

        .icon {
            width: 16px;
            height: 16px;
            stroke-width: 2;
            stroke: currentColor;
            fill: none;
            stroke-linecap: round;
            stroke-linejoin: round;
            flex-shrink: 0;
        }
        .icon-lg {
            width: 24px;
            height: 24px;
        }
        
        .layout {
            display: grid; grid-template-columns: auto 380px; gap: 24px;
            max-width: 1100px; width: 100%; justify-content: center;
        }
        .card {
            background: var(--panel-bg); padding: 20px; border-radius: 12px;
            border: 1px solid #334155; box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        }

        /* Grid Styling */
        #grid-container {
            display: grid; grid-template-columns: repeat(12, 38px);
            grid-template-rows: repeat(12, 38px); gap: 3px;
            margin-top: 12px; user-select: none;
        }
        .cell {
            width: 38px; height: 38px; border-radius: 6px;
            background: #0f172a; border: 1px solid #1e293b;
            display: flex; align-items: center; justify-content: center;
            font-size: 11px; font-weight: bold; cursor: pointer;
            transition: transform 0.1s, background-color 0.2s;
        }
        .cell:hover { transform: scale(1.08); border-color: var(--accent); }
        .cell.firewall { background: var(--firewall); border-color: #475569; }
        .cell.ids { background: rgba(244, 63, 94, 0.25); border: 1px solid var(--ids-trap); color: #fda4af; }
        .cell.start { background: var(--hacker-green) !important; color: #052e16; font-size: 13px; }
        .cell.goal { background: var(--goal-gold) !important; color: #422006; font-size: 13px; }
        .cell.visited { background: var(--visited); border-color: var(--accent); }
        .cell.path { background: var(--path) !important; color: #052e16 !important; transform: scale(1.05); }

        /* Controls & Buttons */
        .toolbar, .algo-btns { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
        button {
            background: #334155; color: white; border: 1px solid #475569;
            padding: 8px 12px; border-radius: 6px; cursor: pointer;
            font-weight: 600; font-size: 13px; transition: 0.15s;
            display: inline-flex; align-items: center; justify-content: center; gap: 6px;
        }
        button:hover { background: #475569; }
        button.active-tool { border-color: var(--accent); background: rgba(56, 189, 248, 0.2); color: var(--accent); }
        button.primary { background: #0284c7; border-color: #38bdf8; }
        button.primary:hover { background: #0369a1; }

        /* Legend */
        .legend { display: flex; gap: 14px; font-size: 12px; color: var(--text-muted); margin-bottom: 10px; flex-wrap: wrap; }
        .legend span { display: flex; align-items: center; gap: 6px; }
        .dot { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }

        /* Table & Telemetry */
        table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 13px; }
        th, td { padding: 9px 8px; text-align: left; border-bottom: 1px solid #334155; }
        th { color: var(--text-muted); font-weight: 600; }
        .badge-safe { background: rgba(34,197,94,0.2); color: #4ade80; padding: 3px 7px; border-radius: 4px; font-size: 11px; }
        .badge-danger { background: rgba(244,63,94,0.2); color: #fb7185; padding: 3px 7px; border-radius: 4px; font-size: 11px; }
        .status-box {
            background: #0f172a; padding: 12px; border-radius: 8px;
            border-left: 4px solid var(--accent); margin-bottom: 16px; font-size: 13px;
        }
    </style>
</head>
<body>

    <h1>
        <svg class="icon icon-lg" viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
        Stealth Hacker Grid: AI Pathfinding
    </h1>
    <p class="sub">Simulasi Penetrasi Jaringan: Perbandingan Uninformed Search (BFS, UCS) vs Informed Search (A*)</p>

    <div class="layout">
        <!-- PANEL KIRI: GRID VISUALIZER -->
        <div class="card">
            <div class="toolbar">
                <span style="font-size:13px; align-self:center; color:#94a3b8; margin-right:4px;">Mode Gambar:</span>
                <button id="tool-1" class="active-tool" onclick="setTool(1)">
                    <svg class="icon" viewBox="0 0 24 24"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/><path d="M3 15h18"/><path d="M9 3v6"/><path d="M15 9v6"/><path d="M9 15v6"/></svg>
                    Firewall (Tembok)
                </button>
                <button id="tool-2" onclick="setTool(2)">
                    <svg class="icon" viewBox="0 0 24 24"><path d="M7 18v-6a5 5 0 1 1 10 0v6"/><path d="M5 21a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-1a2 2 0 0 0-2-2H7a2 2 0 0 0-2 2z"/><line x1="12" y1="2" x2="12" y2="4"/></svg>
                    Sensor IDS (Cost: 10)
                </button>
                <button id="tool-0" onclick="setTool(0)">
                    <svg class="icon" viewBox="0 0 24 24"><rect width="20" height="8" x="2" y="2" rx="2" ry="2"/><rect width="20" height="8" x="2" y="14" rx="2" ry="2"/><line x1="6" x2="6.01" y1="6" y2="6"/><line x1="6" x2="6.01" y1="18" y2="18"/></svg>
                    Server Normal (Cost: 1)
                </button>
                <button onclick="resetDefaultMap()">
                    <svg class="icon" viewBox="0 0 24 24"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>
                    Reset Map Default
                </button>
            </div>

            <div class="legend">
                <span><i class="dot" style="background:var(--hacker-green)"></i> Start (Hacker)</span>
                <span><i class="dot" style="background:var(--goal-gold)"></i> Goal (Main DB)</span>
                <span><i class="dot" style="background:var(--firewall)"></i> Firewall (Blocked)</span>
                <span><i class="dot" style="background:var(--ids-trap)"></i> IDS/Honeypot (Cost 10)</span>
                <span><i class="dot" style="background:var(--visited)"></i> Dieksplorasi AI</span>
            </div>

            <div id="grid-container"></div>
        </div>

        <!-- PANEL KANAN: KONTROL ALGORITMA & TABEL KOMPARASI -->
        <div class="card">
            <h3 style="margin-top:0; font-size:16px;">Jalankan & Bandingkan Model</h3>
            <div class="algo-btns">
                <button class="primary" onclick="runSolver('bfs')">
                    <svg class="icon" viewBox="0 0 24 24" fill="currentColor"><polygon points="6 3 20 12 6 21 6 3"/></svg>
                    Jalankan BFS
                </button>
                <button class="primary" onclick="runSolver('ucs')">
                    <svg class="icon" viewBox="0 0 24 24" fill="currentColor"><polygon points="6 3 20 12 6 21 6 3"/></svg>
                    Jalankan UCS
                </button>
                <button class="primary" onclick="runSolver('astar')">
                    <svg class="icon" viewBox="0 0 24 24" fill="currentColor"><polygon points="6 3 20 12 6 21 6 3"/></svg>
                    Jalankan A*
                </button>
            </div>

            <div id="status" class="status-box">
                Klik salah satu tombol di atas untuk melihat animasi pencarian jalur penetrasi jaringan dan perbandingan ke-3 model.
            </div>

            <h4 style="margin: 16px 0 6px 0; font-size:14px; color:var(--accent); display:flex; align-items:center; gap:6px;">
                <svg class="icon" viewBox="0 0 24 24"><path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/></svg>
                Tabel Komparasi Eksperimen (Bab 3)
            </h4>
            <table>
                <thead>
                    <tr>
                        <th>Algoritma</th>
                        <th>Langkah</th>
                        <th>Risk Cost</th>
                        <th>Node</th>
                        <th>Waktu</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody id="result-tbody">
                    <tr><td colspan="6" style="text-align:center; color:#64748b;">Belum dijalankan</td></tr>
                </tbody>
            </table>

            <div style="margin-top:18px; font-size:12px; color:var(--text-muted); line-height:1.5;">
                <strong style="display:flex; align-items:center; gap:5px; color:var(--text-main); margin-bottom:4px;">
                    <svg class="icon" style="color:#eab308;" viewBox="0 0 24 24"><path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/></svg>
                    Analisis Otomatis:
                </strong>
                • <b>BFS</b> mencari jumlah langkah (hop) paling sedikit tanpa peduli bobot risiko, sehingga menerobos sensor <b>IDS</b> dan memicu alarm.<br>
                • <b>UCS</b> memutar mencari jalur stealth termurah, tetapi mengeksplorasi terlalu banyak node.<br>
                • <b>A*</b> menemukan jalur stealth yang sama optimalnya dengan UCS, namun dengan jumlah <i>expanded nodes</i> jauh lebih sedikit berkat <i>Manhattan Heuristic</i>.
            </div>
        </div>
    </div>

<script>
    const ROWS = 12, COLS = 12;
    const START = [1, 1], GOAL = [10, 10];
    let currentTool = 1;
    let gridData = [];
    let animTimers = [];

    // Preset Map yang menonjolkan perbedaan BFS vs UCS vs A*
    const DEFAULT_MAP = [
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,1,1,1,1,0,0,0,0,0],
        [0,0,0,0,0,0,1,0,1,1,0,0],
        [0,1,1,2,2,0,1,0,0,1,0,0],
        [0,0,1,2,2,2,1,1,0,1,0,0],
        [0,0,0,2,2,2,2,0,0,0,0,0],
        [0,1,1,1,2,2,2,2,1,1,0,0],
        [0,0,0,1,1,2,2,2,2,1,0,0],
        [0,1,0,0,0,2,2,2,2,0,0,0],
        [0,1,0,1,1,1,2,2,0,0,0,0],
        [0,0,0,0,0,0,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0]
    ];

    function setTool(t) {
        currentTool = t;
        document.querySelectorAll('.toolbar button').forEach(b => b.classList.remove('active-tool'));
        const btn = document.getElementById('tool-' + t);
        if (btn) btn.classList.add('active-tool');
    }

    function clearTimers() {
        animTimers.forEach(t => clearTimeout(t));
        animTimers = [];
    }

    function renderGrid() {
        const container = document.getElementById('grid-container');
        container.innerHTML = '';
        for (let r = 0; r < ROWS; r++) {
            for (let c = 0; c < COLS; c++) {
                const div = document.createElement('div');
                div.id = `cell-${r}-${c}`;
                div.className = 'cell';

                if (r === START[0] && c === START[1]) {
                    div.classList.add('start');
                    div.textContent = 'HCK';
                } else if (r === GOAL[0] && c === GOAL[1]) {
                    div.classList.add('goal');
                    div.textContent = 'DB';
                } else if (gridData[r][c] === 1) {
                    div.classList.add('firewall');
                } else if (gridData[r][c] === 2) {
                    div.classList.add('ids');
                    div.textContent = '10';
                } else {
                    div.textContent = '1';
                    div.style.color = '#334155';
                }

                div.onclick = () => {
                    if ((r === START[0] && c === START[1]) || (r === GOAL[0] && c === GOAL[1])) return;
                    gridData[r][c] = currentTool;
                    renderGrid();
                };
                container.appendChild(div);
            }
        }
    }

    function resetDefaultMap() {
        clearTimers();
        gridData = DEFAULT_MAP.map(row => [...row]);
        renderGrid();
    }

    async function runSolver(selectedAlgo) {
        clearTimers();
        renderGrid();

        const res = await fetch('/solve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ grid: gridData, start: START, goal: GOAL })
        });
        const data = await res.json();
        const chosen = data[selectedAlgo];

        // Update Tabel Komparasi Ketiga Model Sekaligus
        const tbody = document.getElementById('result-tbody');
        tbody.innerHTML = '';
        ['bfs', 'ucs', 'astar'].forEach(key => {
            const item = data[key];
            const isDetected = item.cost >= 25;
            const badge = isDetected 
                ? `<span class="badge-danger">Terdeteksi IDS</span>` 
                : `<span class="badge-safe">Stealth Aman</span>`;
            const highlight = key === selectedAlgo ? 'style="background:rgba(56,189,248,0.12); font-weight:bold;"' : '';
            tbody.innerHTML += `
                <tr ${highlight}>
                    <td>${key.toUpperCase()}</td>
                    <td>${item.steps} hop</td>
                    <td>${item.cost}</td>
                    <td>${item.expanded}</td>
                    <td>${item.time_ms} ms</td>
                    <td>${badge}</td>
                </tr>
            `;
        });

        document.getElementById('status').innerHTML = 
            `Menampilkan visualisasi <b>${chosen.name}</b><br>` +
            `Total Risk Cost: <b>${chosen.cost}</b> | Node Dieksplorasi: <b>${chosen.expanded} node</b>`;

        // Animasi Node yang Dieksplorasi (Visited)
        chosen.visited.forEach((coord, idx) => {
            const t = setTimeout(() => {
                const el = document.getElementById(`cell-${coord[0]}-${coord[1]}`);
                if (el && !el.classList.contains('start') && !el.classList.contains('goal')) {
                    el.classList.add('visited');
                }
            }, idx * 18);
            animTimers.push(t);
        });

        // Animasi Jalur Akhir (Path)
        const pathDelay = chosen.visited.length * 18 + 100;
        chosen.path.forEach((coord, idx) => {
            const t = setTimeout(() => {
                const el = document.getElementById(`cell-${coord[0]}-${coord[1]}`);
                if (el && !el.classList.contains('start') && !el.classList.contains('goal')) {
                    el.classList.add('path');
                }
            }, pathDelay + idx * 45);
            animTimers.push(t);
        });
    }

    resetDefaultMap();
</script>
</body>
</html>
"""

# ==========================================
# BAGIAN 3: ROUTING FLASK
# ==========================================
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/solve", methods=["POST"])
def solve():
    req = request.get_json()
    grid = req["grid"]
    start = tuple(req["start"])
    goal = tuple(req["goal"])

    return jsonify({
        "bfs": solve_bfs(grid, start, goal),
        "ucs": solve_ucs(grid, start, goal),
        "astar": solve_astar(grid, start, goal)
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)