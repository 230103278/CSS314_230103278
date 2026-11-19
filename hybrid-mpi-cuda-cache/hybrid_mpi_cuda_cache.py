import os
import re
import sys
import json
import math
import time
import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from statistics import median
from types import SimpleNamespace

import numpy as np
from mpi4py import MPI
from numba import cuda, njit, float64, uint64

TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2  # 18x18 shared tile
GLOBAL_NX = 512
GLOBAL_NY = 512
STENCIL_STEPS = 300
SHM_ITERS = 40_000_000
LAMBDA_DAMP = 0.15

PROC_DIMS = [2, 2]
NY_LOCAL = GLOBAL_NY // PROC_DIMS[0]  # 256 owned rows per rank
NX_LOCAL = GLOBAL_NX // PROC_DIMS[1]  # 256 owned columns per rank
DX = 1.0 / (GLOBAL_NX + 1)  # 1/513: owned cells are global 1..512, Dirichlet walls sit at 0 and 513
DY = 1.0 / (GLOBAL_NY + 1)
CFL_SAFETY = 0.8  # dt = 0.8 * dx^2 / (4 * kappa_max)
REPORT_EVERY = 100
PACK_THREADS = 128

SHM_SLOTS = 64
SHM_STRIDES = (1, 8, 16)
SHM_REPEATS = 3  # median of 3 timed runs per stride
SHM_ALIGN = 128  # satisfies the 64-byte requirement and also aligns 128-byte adjacent-line pairs

# Segment order inside the 1D halo buffers. A strip sent towards direction d carries tag d,
# so the receiver expects the tag of the opposite direction.
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
OPPOSITE = (DOWN, UP, RIGHT, LEFT)


def _abort_on_error(exc_type, exc, tb):
    sys.__excepthook__(exc_type, exc, tb)
    MPI.COMM_WORLD.Abort(1)  # one crashed rank must not leave the other three blocked in a collective


sys.excepthook = _abort_on_error


def get_gpu_uuid() -> str:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        encoding="utf-8",
    )
    return out.strip().splitlines()[0].strip()


def detect_cpu_line_size() -> int:
    path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    return 64


def read_cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def physical_core_cpus():
    """First logical CPU of every physical core this process may run on."""
    if not hasattr(os, "sched_getaffinity"):
        return []
    chosen, seen = [], set()
    for cpu in sorted(os.sched_getaffinity(0)):
        try:
            with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list", encoding="utf-8") as f:
                core = f.read().strip()
        except OSError:
            core = str(cpu)
        if core not in seen:
            seen.add(core)
            chosen.append(cpu)
    return chosen


# ==============================================================================
# TASK 1: Hybrid MPI Cartesian Grid + Node-Local MPI-3 Shared-Memory Window
# ==============================================================================
def init_hybrid_context(github_user: str) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f"Requires exactly 4 MPI ranks (2x2 grid), got {size}.")

    cuda.select_device(0)
    gpu_uuid = comm.bcast(get_gpu_uuid() if rank == 0 else None, root=0)
    cpu_sig = f"{platform.machine()}::{os.cpu_count()}"
    line_bytes = detect_cpu_line_size()

    digest = hashlib.sha256(f"{github_user.strip()}::{gpu_uuid}::{cpu_sig}".encode("utf-8")).hexdigest()
    instance_seed = int(digest[:16], 16)

    cart = comm.Create_cart(dims=PROC_DIMS, periods=[False, False], reorder=False)
    cart_rank = cart.Get_rank()
    nbr_up, nbr_down = cart.Shift(0, 1)  # axis 0 = rows (y)
    nbr_left, nbr_right = cart.Shift(1, 1)  # axis 1 = columns (x)

    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    if node_comm.Get_size() != size:
        raise RuntimeError("All 4 ranks must share one node for the MPI-3 shared-memory window.")
    node_rank = node_comm.Get_rank()
    nbytes = SHM_SLOTS * 8 + SHM_ALIGN
    win = MPI.Win.Allocate_shared(nbytes if node_rank == 0 else 0, 1, comm=node_comm)
    buf, _ = win.Shared_query(0)
    raw = np.ndarray(buffer=buf, dtype=np.uint8, shape=(nbytes,))
    # Every process maps the segment at a page boundary, so one offset aligns it in all of them.
    offset = node_comm.bcast((-raw.ctypes.data) % SHM_ALIGN if node_rank == 0 else None, root=0)
    shm_view = raw[offset:offset + SHM_SLOTS * 8].view(np.uint64)
    if shm_view.ctypes.data % 64 != 0:
        raise RuntimeError("Shared window slot 0 is not 64-byte aligned on this rank.")
    win.Lock_all(MPI.MODE_NOCHECK)  # passive-target epoch so win.Sync() orders the shared loads/stores

    return {
        "comm": comm,
        "cart": cart,
        "rank": cart_rank,
        "coords": tuple(cart.Get_coords(cart_rank)),
        "nbrs": (nbr_up, nbr_down, nbr_left, nbr_right),
        "node_comm": node_comm,
        "shm_win": win,
        "shm_view": shm_view,
        "gpu_uuid": gpu_uuid,
        "cpu_sig": cpu_sig,
        "cpu_model": read_cpu_model(),
        "line_bytes": line_bytes,
        "seed": instance_seed,
    }


# ==============================================================================
# TASK 2: Rank-Local GPU Subdomain & Halo Pack/Unpack Kernels
# ==============================================================================
def initialize_fields(ctx):
    row_rank, col_rank = ctx["coords"]
    # Same seed on every rank -> every rank draws identical pulse parameters -> one consistent global field.
    rng = np.random.default_rng(ctx["seed"])
    n_pulses = 3
    pulse_x = rng.uniform(0.2, 0.8, n_pulses)
    pulse_y = rng.uniform(0.2, 0.8, n_pulses)
    pulse_amp = rng.uniform(1.0, 2.0, n_pulses)
    pulse_sigma = rng.uniform(0.04, 0.08, n_pulses)
    phase_x, phase_y = rng.uniform(0.0, 2.0 * np.pi, 2)
    src_x, src_y = rng.uniform(0.3, 0.7, 2)

    # Local index l = 0..257 is global index rank_coord * 256 + l, so x = gj * dx covers [0, 1].
    y = (row_rank * NY_LOCAL + np.arange(NY_LOCAL + 2)) * DY
    x = (col_rank * NX_LOCAL + np.arange(NX_LOCAL + 2)) * DX
    X, Y = np.meshgrid(x, y)  # (258, 258): [i, j] -> (y_i, x_j)

    kappa = 0.16 + 0.08 * np.sin(2.0 * np.pi * X + phase_x) * np.cos(2.0 * np.pi * Y + phase_y)
    u = np.zeros_like(X)
    for cx, cy, amp, sig in zip(pulse_x, pulse_y, pulse_amp, pulse_sigma):
        u += amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sig * sig))
    src = 0.05 * np.exp(-((X - src_x) ** 2 + (Y - src_y) ** 2) / (2.0 * 0.03**2))

    nbr_up, nbr_down, nbr_left, nbr_right = ctx["nbrs"]
    if nbr_up == MPI.PROC_NULL:
        u[0, :] = 0.0
    if nbr_down == MPI.PROC_NULL:
        u[-1, :] = 0.0
    if nbr_left == MPI.PROC_NULL:
        u[:, 0] = 0.0
    if nbr_right == MPI.PROC_NULL:
        u[:, -1] = 0.0
    return u, kappa, src


@cuda.jit
def pack_halos_kernel(u, send, n):
    k = cuda.grid(1)
    if k < n:
        send[UP * n + k] = u[1, k + 1]  # first owned row
        send[DOWN * n + k] = u[n, k + 1]  # last owned row
        send[LEFT * n + k] = u[k + 1, 1]  # first owned column (stride 258 doubles)
        send[RIGHT * n + k] = u[k + 1, n]  # last owned column


@cuda.jit
def unpack_halos_kernel(u, recv, n):
    k = cuda.grid(1)
    if k < n:
        u[0, k + 1] = recv[UP * n + k]
        u[n + 1, k + 1] = recv[DOWN * n + k]
        u[k + 1, 0] = recv[LEFT * n + k]
        u[k + 1, n + 1] = recv[RIGHT * n + k]


def exchange_halos_async(st, u, stream):
    """Pack on the GPU -> async D2H into pinned memory -> post tag-matched Irecv/Isend."""
    n = st.n
    pack_halos_kernel[st.pack_blocks, PACK_THREADS, stream](u, st.d_send, n)
    st.d_send.copy_to_host(st.h_send, stream=stream)
    stream.synchronize()  # MPI reads h_send from the host, so the D2H DMA must have landed
    reqs = [
        st.cart.Irecv([st.h_recv[d * n:(d + 1) * n], MPI.DOUBLE], source=st.nbrs[d], tag=OPPOSITE[d])
        for d in range(4)
    ]
    reqs += [
        st.cart.Isend([st.h_send[d * n:(d + 1) * n], MPI.DOUBLE], dest=st.nbrs[d], tag=d)
        for d in range(4)
    ]
    return reqs


def complete_halos(st, u, reqs, stream):
    """Waitall -> async H2D from pinned memory -> GPU unpack into ghosts 0 and 257.

    A PROC_NULL receive never touches its zero-filled segment, so wall ghosts are re-written with the
    Dirichlet value u = 0 without any boundary branch.
    """
    MPI.Request.Waitall(reqs)
    st.d_recv.copy_to_device(st.h_recv, stream=stream)
    unpack_halos_kernel[st.pack_blocks, PACK_THREADS, stream](u, st.d_recv, st.n)


# ==============================================================================
# TASK 3: 2D Shared-Memory Tiled Stencil & Dual-Stream Pipeline
# ==============================================================================
@cuda.jit
def tiled_stencil_kernel(u, u_new, kappa, src, r_start, r_end, c_start, c_end, dt, inv_dx2, inv_dy2, lam):
    tile = cuda.shared.array(shape=(SHARED_DIM, SHARED_DIM), dtype=float64)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    i = r_start + cuda.blockIdx.y * TILE_DIM + ty
    j = c_start + cuda.blockIdx.x * TILE_DIM + tx
    sy = ty + 1
    sx = tx + 1

    # Stage only the box plus its 1-cell apron: the deep-interior launch then never reads ghost
    # cells, which stream_boundary may be unpacking at the same moment.
    row_ok = i <= r_end + 1
    col_ok = j <= c_end + 1
    if row_ok and col_ok:
        tile[sy, sx] = u[i, j]
        if ty == 0:
            tile[0, sx] = u[i - 1, j]
        if tx == 0:
            tile[sy, 0] = u[i, j - 1]
    if ty == TILE_DIM - 1 and i <= r_end and col_ok:
        tile[sy + 1, sx] = u[i + 1, j]
    if tx == TILE_DIM - 1 and j <= c_end and row_ok:
        tile[sy, sx + 1] = u[i, j + 1]
    cuda.syncthreads()

    if i <= r_end and j <= c_end:
        uc = tile[sy, sx]
        kc = kappa[i, j]
        div_y = (0.5 * (kappa[i + 1, j] + kc) * (tile[sy + 1, sx] - uc)
                 - 0.5 * (kappa[i - 1, j] + kc) * (uc - tile[sy - 1, sx])) * inv_dy2
        div_x = (0.5 * (kappa[i, j + 1] + kc) * (tile[sy, sx + 1] - uc)
                 - 0.5 * (kappa[i, j - 1] + kc) * (uc - tile[sy, sx - 1])) * inv_dx2
        u_new[i, j] = uc + dt * (div_x + div_y - lam * uc * uc * uc + src[i, j])


def launch_stencil(st, u, u_new, box, stream):
    r_start, r_end, c_start, c_end = box
    grid = ((c_end - c_start + TILE_DIM) // TILE_DIM, (r_end - r_start + TILE_DIM) // TILE_DIM)
    tiled_stencil_kernel[grid, (TILE_DIM, TILE_DIM), stream](
        u, u_new, st.kappa, st.src, r_start, r_end, c_start, c_end, st.dt, st.inv_dx2, st.inv_dy2, LAMBDA_DAMP
    )


def step_serialized(st, u, u_new):
    """Reference: full halo exchange first, then one stencil over all owned cells, single stream."""
    reqs = exchange_halos_async(st, u, st.stream_boundary)
    complete_halos(st, u, reqs, st.stream_boundary)
    launch_stencil(st, u, u_new, st.full_box, st.stream_boundary)
    st.stream_boundary.synchronize()


def step_overlapped(st, u, u_new):
    reqs = exchange_halos_async(st, u, st.stream_boundary)  # pack + D2H + post Irecv/Isend
    launch_stencil(st, u, u_new, st.interior_box, st.stream_interior)  # [2..255]^2 needs no ghosts
    complete_halos(st, u, reqs, st.stream_boundary)  # Waitall + H2D + unpack
    for box in st.ring_boxes:
        launch_stencil(st, u, u_new, box, st.stream_boundary)
    st.stream_interior.synchronize()
    st.stream_boundary.synchronize()


def global_diagnostics(cart, u, u_prev):
    """Global energy sum(u)*dx*dy and L2 residual ||u^n - u^(n-1)|| via cart_comm.allreduce."""
    a = u.copy_to_host()[1:-1, 1:-1]
    b = u_prev.copy_to_host()[1:-1, 1:-1]
    energy = cart.allreduce(float(a.sum()) * DX * DY, op=MPI.SUM)
    res2 = cart.allreduce(float(((a - b) ** 2).sum()) * DX * DY, op=MPI.SUM)
    return energy, math.sqrt(res2)


def run_mode(st, step_fn, steps):
    u = cuda.to_device(st.u0)
    u_new = cuda.to_device(st.u0)  # wall ghosts start at 0 in both buffers; the stencil never writes ghosts
    trace = [(0, *global_diagnostics(st.cart, u, u_new))]
    start, stop = cuda.event(), cuda.event()
    st.cart.Barrier()
    start.record(st.stream_boundary)
    for n in range(1, steps + 1):
        step_fn(st, u, u_new)
        u, u_new = u_new, u
        if n % REPORT_EVERY == 0:
            trace.append((n, *global_diagnostics(st.cart, u, u_new)))
    stop.record(st.stream_boundary)
    stop.synchronize()
    ms = st.cart.allreduce(cuda.event_elapsed_time(start, stop), op=MPI.MAX)
    return u.copy_to_host(), ms, trace


def run_gpu_stencil(ctx) -> dict:
    cart = ctx["cart"]
    u0, kappa, src = initialize_fields(ctx)
    kappa_max = cart.allreduce(float(kappa[1:-1, 1:-1].max()), op=MPI.MAX)
    kappa_min = cart.allreduce(float(kappa[1:-1, 1:-1].min()), op=MPI.MIN)
    dt_max = DX * DX / (4.0 * kappa_max)
    dt = CFL_SAFETY * dt_max

    n = NX_LOCAL
    st = SimpleNamespace(
        cart=cart,
        nbrs=ctx["nbrs"],
        n=n,
        u0=u0,
        dt=dt,
        inv_dx2=1.0 / (DX * DX),
        inv_dy2=1.0 / (DY * DY),
        kappa=cuda.to_device(kappa),
        src=cuda.to_device(src),
        d_send=cuda.device_array(4 * n, dtype=np.float64),
        d_recv=cuda.device_array(4 * n, dtype=np.float64),
        h_send=cuda.pinned_array(4 * n, dtype=np.float64),
        h_recv=cuda.pinned_array(4 * n, dtype=np.float64),
        stream_boundary=cuda.stream(),
        stream_interior=cuda.stream(),
        pack_blocks=(n + PACK_THREADS - 1) // PACK_THREADS,
        full_box=(1, n, 1, n),
        interior_box=(2, n - 1, 2, n - 1),
        ring_boxes=((1, 1, 1, n), (n, n, 1, n), (2, n - 1, 1, 1), (2, n - 1, n, n)),
    )
    st.h_send[:] = 0.0
    st.h_recv[:] = 0.0  # pinned memory starts uninitialised; PROC_NULL segments must read as u = 0

    run_mode(st, step_serialized, 1)  # JIT warm-up so compilation is not timed
    run_mode(st, step_overlapped, 1)
    ser_u, ser_ms, ser_trace = run_mode(st, step_serialized, STENCIL_STEPS)
    ovl_u, ovl_ms, ovl_trace = run_mode(st, step_overlapped, STENCIL_STEPS)
    parity = cart.allreduce(float(np.abs(ser_u - ovl_u).max()), op=MPI.MAX)
    energies = [e for _, e, _ in ovl_trace]
    return {
        "kappa_min": kappa_min,
        "kappa_max": kappa_max,
        "dt_max_cfl": dt_max,
        "dt": dt,
        "ser_trace": ser_trace,
        "ovl_trace": ovl_trace,
        "gpu_serialized_ms": ser_ms,
        "gpu_overlapped_ms": ovl_ms,
        "parity_max_diff": parity,
        "energy_monotonic_decrease": all(b < a for a, b in zip(energies, energies[1:])),
    }


# ==============================================================================
# TASK 4: Multi-Rank Host Cache-Line False Sharing Sweep (MPI-3 shared window)
# ==============================================================================
@njit(nogil=True)
def host_shm_accumulate(shm_view, slot, iters, seed):
    inc = uint64(seed % 7 + 1)
    for _ in range(iters):
        v = shm_view[slot]
        # v never reaches 2**63, so (v >> 63) is 0 at run time, but LLVM cannot prove it. The store
        # address therefore depends on the loaded value, which blocks register promotion (LICM):
        # every iteration really loads and stores shm_view[slot] in the shared window.
        shm_view[slot + np.int64(v >> uint64(63))] = v + inc


def shm_fence(ctx):
    ctx["shm_win"].Sync()
    ctx["node_comm"].Barrier()
    ctx["shm_win"].Sync()


def benchmark_shm_strides(ctx) -> dict:
    nc, view = ctx["node_comm"], ctx["shm_view"]
    r, size = nc.Get_rank(), nc.Get_size()
    seed = (ctx["seed"] + r * 0x9E3779B97F4A7C15) % (1 << 63)  # < 2**63 so Numba types it as int64
    expected = SHM_ITERS * (seed % 7 + 1)

    cores = physical_core_cpus()
    pinned = nc.allreduce(len(cores) >= size, op=MPI.LAND)
    if pinned:
        os.sched_setaffinity(0, {cores[r]})  # one rank per physical core: SMT siblings would share an L1

    host_shm_accumulate(view, SHM_SLOTS - 8 + r, 10, seed)  # JIT warm-up on a scratch slot
    results = {}
    for stride in SHM_STRIDES:
        slot = r * stride
        runs, all_ok = [], True
        for _ in range(SHM_REPEATS):
            if r == 0:
                view[:] = 0
            shm_fence(ctx)
            t0 = time.perf_counter_ns()  # CLOCK_MONOTONIC: one clock shared by all ranks on the node
            host_shm_accumulate(view, slot, SHM_ITERS, seed)
            t1 = time.perf_counter_ns()
            shm_fence(ctx)
            ok = int(view[slot]) == expected
            all_ok = nc.allreduce(ok, op=MPI.LAND) and all_ok
            runs.append(nc.allreduce(t1, op=MPI.MAX) - nc.allreduce(t0, op=MPI.MIN))
        med = int(median(runs))
        results[stride] = {
            "slot_byte_offsets": [8 * k * stride for k in range(size)],
            "cache_lines_64B": [8 * k * stride // 64 for k in range(size)],
            "line_pairs_128B": [8 * k * stride // 128 for k in range(size)],
            "runs_ns": runs,
            "median_ns": med,
            "ns_per_update": med / SHM_ITERS,
            "counters_ok": all_ok,
        }
    return {
        "window_slots": SHM_SLOTS,
        "alignment_bytes": SHM_ALIGN,
        "iters_per_rank": SHM_ITERS,
        "repeats": SHM_REPEATS,
        "pinned_to_physical_cores": bool(pinned),
        "physical_cores_available": len(cores),
        "strides": results,
        "slowdown_stride1_vs_stride8": results[1]["median_ns"] / results[8]["median_ns"],
        "ratio_stride16_vs_stride8": results[16]["median_ns"] / results[8]["median_ns"],
    }


# ==============================================================================
# TASK 5: Cross-Tier Verification & Manifest Export
# ==============================================================================
def ptx_audit():
    asm = tiled_stencil_kernel.inspect_asm()
    ptx = "\n".join(asm.values()) if isinstance(asm, dict) else str(asm)
    counts = {op: ptx.count(op) for op in ("st.shared.f64", "ld.shared.f64", "bar.sync")}
    return counts, all(c > 0 for c in counts.values())


def llvm_audit(dispatcher) -> dict:
    ir = "\n".join(dispatcher.inspect_llvm().values())
    functions = re.split(r"\n(?=define )", ir)
    name = dispatcher.py_func.__name__
    # the native function, not Numba's "cpython" wrapper that boxes arguments
    core = [f for f in functions if f.startswith("define") and name in f.split("(")[0]
            and "cpython" not in f.split("(")[0]]
    body = "\n".join(core)
    return {
        "native_functions": len(core),
        "store_i64": len(re.findall(r"\bstore i64\b", body)),
        "load_i64": len(re.findall(r"= load i64\b", body)),
        "python_api_calls": len(re.findall(r"@_?Py\w+", body)),
    }


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: mpirun -np 4 python3 hybrid_mpi_cuda_cache.py <github_username>")
    github_user = sys.argv[1]
    ctx = init_hybrid_context(github_user)
    cart, rank = ctx["cart"], ctx["rank"]

    if rank == 0:
        dev = cuda.get_current_device()
        print("=" * 78)
        print(f"GitHub user      : {github_user}")
        print(f"GPU UUID         : {ctx['gpu_uuid']}")
        print(f"GPU              : CC {dev.compute_capability}, {dev.MULTIPROCESSOR_COUNT} SMs, "
              f"{dev.ASYNC_ENGINE_COUNT} async copy engines")
        print(f"CPU              : {ctx['cpu_model']} | cpu_sig={ctx['cpu_sig']} | "
              f"{len(physical_core_cpus())} physical cores | line {ctx['line_bytes']} B")
        print(f"INSTANCE_SEED    : {ctx['seed']} (0x{ctx['seed']:016x})")
        print(f"MPI ranks        : {cart.Get_size()} in a {PROC_DIMS[0]}x{PROC_DIMS[1]} Cartesian grid, "
              f"periods=[False, False], MPI.PROC_NULL={MPI.PROC_NULL}, node_comm size={ctx['node_comm'].Get_size()}")
        print("=" * 78, flush=True)
    for r in range(cart.Get_size()):
        if r == rank:
            up, down, left, right = ctx["nbrs"]
            print(f"[rank {rank}] coords={ctx['coords']} up={up} down={down} left={left} right={right} "
                  f"local=(258, 258) shm slot0 addr mod 64 = {ctx['shm_view'].ctypes.data % 64}", flush=True)
        cart.Barrier()

    gpu = run_gpu_stencil(ctx)
    shm = benchmark_shm_strides(ctx)
    ptx_counts, ptx_ok = ptx_audit()
    llvm = llvm_audit(host_shm_accumulate)
    llvm_ok = llvm["native_functions"] >= 1 and llvm["store_i64"] >= 1 and llvm["python_api_calls"] == 0
    parity_ok = gpu["parity_max_diff"] < 1e-12
    shm_ok = all(s["counters_ok"] for s in shm["strides"].values())
    passed = parity_ok and ptx_ok and llvm_ok and shm_ok

    if rank == 0:
        print(f"\ndx = dy = 1/{GLOBAL_NX + 1} = {DX:.9e} | kappa range [{gpu['kappa_min']:.6f}, {gpu['kappa_max']:.6f}]")
        print(f"dt_max (CFL) = dx^2/(4 kappa_max) = {gpu['dt_max_cfl']:.6e} | dt = {CFL_SAFETY} * dt_max = {gpu['dt']:.6e}")
        for name, trace in (("serialized", gpu["ser_trace"]), ("overlapped", gpu["ovl_trace"])):
            print(f"\ncart_comm.allreduce diagnostics ({name}):")
            print(f"{'step':>6} {'energy E':>22} {'L2 residual':>22}")
            for s, e, res in trace:
                print(f"{s:>6} {e:>22.15e} {res:>22.15e}")
        g_ser, g_ovl = gpu["gpu_serialized_ms"], gpu["gpu_overlapped_ms"]
        print(f"\ngpu_serialized_ms : {g_ser:.3f} ({g_ser / STENCIL_STEPS:.4f} ms/step)")
        print(f"gpu_overlapped_ms : {g_ovl:.3f} ({g_ovl / STENCIL_STEPS:.4f} ms/step) -> {g_ser / g_ovl:.3f}x")
        print(f"parity_max_diff   : {gpu['parity_max_diff']:.3e} -> {'PASS' if parity_ok else 'FAIL'} (< 1e-12)")
        print(f"PTX audit         : {ptx_counts} -> {'PASS' if ptx_ok else 'FAIL'}")

        print(f"\nMPI-3 shared window: {SHM_SLOTS} uint64 slots, {SHM_ALIGN}-B aligned, "
              f"{SHM_ITERS:,} updates/rank, median of {SHM_REPEATS}, "
              f"pinned to physical cores: {shm['pinned_to_physical_cores']}")
        print(f"{'stride':>6} {'64B lines':>14} {'median ns':>16} {'ns/update':>10} {'vs s=8':>8} counters")
        s8 = shm["strides"][8]["median_ns"]
        for stride, s in shm["strides"].items():
            print(f"{stride:>6} {str(s['cache_lines_64B']):>14} {s['median_ns']:>16,} {s['ns_per_update']:>10.3f} "
                  f"{s['median_ns'] / s8:>8.2f} {'OK' if s['counters_ok'] else 'FAIL'}")
        print(f"slowdown stride=1 vs stride=8 : {shm['slowdown_stride1_vs_stride8']:.2f}x")
        print(f"stride=16 / stride=8          : {shm['ratio_stride16_vs_stride8']:.3f}")
        print(f"LLVM audit (host_shm_accumulate): {llvm} -> {'PASS' if llvm_ok else 'FAIL'}")

        manifest = {
            "github_username": github_user,
            "gpu_uuid": ctx["gpu_uuid"],
            "cpu_sig": ctx["cpu_sig"],
            "cpu_model": ctx["cpu_model"],
            "cpu_line_bytes": ctx["line_bytes"],
            "instance_seed": ctx["seed"],
            "mpi_ranks": cart.Get_size(),
            "cart_dims": PROC_DIMS,
            "cart_periods": [False, False],
            "node_comm_size": ctx["node_comm"].Get_size(),
            "global_interior_shape": [GLOBAL_NY, GLOBAL_NX],
            "local_padded_shape": [NY_LOCAL + 2, NX_LOCAL + 2],
            "stencil_steps": STENCIL_STEPS,
            "dx": DX,
            "dt": gpu["dt"],
            "kappa_min": gpu["kappa_min"],
            "kappa_max": gpu["kappa_max"],
            "dt_max_cfl": gpu["dt_max_cfl"],
            "lambda_damp": LAMBDA_DAMP,
            "energy_trace": [{"step": s, "energy": e, "l2_residual": res} for s, e, res in gpu["ovl_trace"]],
            "energy_monotonic_decrease": gpu["energy_monotonic_decrease"],
            "gpu_serialized_ms": gpu["gpu_serialized_ms"],
            "gpu_overlapped_ms": gpu["gpu_overlapped_ms"],
            "gpu_overlap_speedup": gpu["gpu_serialized_ms"] / gpu["gpu_overlapped_ms"],
            "parity_max_diff": gpu["parity_max_diff"],
            "parity_passed": parity_ok,
            "ptx_counts": ptx_counts,
            "ptx_audit_passed": ptx_ok,
            "shm_benchmark": {**shm, "strides": {str(k): v for k, v in shm["strides"].items()}},
            "llvm_audit": llvm,
            "llvm_audit_passed": llvm_ok,
            "verification_passed": passed,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }
        with open("hybrid_lab_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nverification_passed: {str(passed).lower()} -> wrote hybrid_lab_manifest.json", flush=True)

    ctx["shm_win"].Unlock_all()
    ctx["shm_win"].Free()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
