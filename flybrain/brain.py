"""GPU leaky integrate-and-fire model of the whole Drosophila brain.

A PyTorch port of the Brian2 model from Shiu et al. 2024 (Nature),
"A Drosophila computational brain model reveals sensorimotor processing".
Same equations and parameters, but all ~139k neurons are stepped together
on the GPU so the brain can run in a closed loop with a simulated body.

    dv/dt = (v_0 - v + g) / t_mbr      (membrane)
    dg/dt = -g / tau                   (synaptic conductance, in volts)
    presynaptic spike -> g_post += w   after a delay of t_dly
"""
from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import pandas as pd
import torch

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class BrainParams:
    dt: float = 0.1          # ms, integration step
    v_0: float = -52.0       # mV, resting potential
    v_rst: float = -52.0     # mV, reset potential
    v_th: float = -45.0      # mV, spike threshold
    t_mbr: float = 20.0      # ms, membrane time constant
    tau: float = 5.0         # ms, synaptic time constant
    t_rfc: float = 2.2       # ms, refractory period
    t_dly: float = 1.8       # ms, synaptic delay
    w_syn: float = 0.275     # mV per synapse
    f_poi: float = 250.0     # Poisson input weight = w_syn * f_poi



class Brain:
    """Whole-brain LIF network. Runs `batch` independent copies in parallel."""

    def __init__(self, params: BrainParams = BrainParams(), batch: int = 1,
                 device: str | None = None, seed: int | None = None):
        self.p = params
        self.batch = batch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.gen = torch.Generator(device=self.device)
        if seed is not None:
            self.gen.manual_seed(seed)

        comp = pd.read_csv(DATA_DIR / "Completeness_783.csv", index_col=0)
        con = pd.read_parquet(DATA_DIR / "Connectivity_783.parquet")
        self.flyids = comp.index.to_numpy()
        self.flyid2i = {f: i for i, f in enumerate(self.flyids)}
        self.n = n = len(self.flyids)

        # W[post, pre] in mV; signed synapse count (negative = inhibitory)
        pre = torch.as_tensor(con["Presynaptic_Index"].to_numpy(), dtype=torch.int64)
        post = torch.as_tensor(con["Postsynaptic_Index"].to_numpy(), dtype=torch.int64)
        w = torch.as_tensor(con["Excitatory x Connectivity"].to_numpy(), dtype=torch.float32)
        self.W = torch.sparse_coo_tensor(
            torch.stack([post, pre]), w * params.w_syn, (n, n)
        ).coalesce().to_sparse_csr().to(self.device)
        # the same weights indexed by presynaptic neuron, for event-driven propagation:
        # only a few dozen of the ~139k neurons spike per step, so gathering their
        # outgoing synapses is far cheaper than a full sparse matrix product
        WT = torch.sparse_coo_tensor(torch.stack([pre, post]), w * params.w_syn, (n, n)).coalesce().to_sparse_csr()
        self.out_ptr = WT.crow_indices().to(self.device)
        self.out_post = WT.col_indices().to(self.device)
        self.out_w = WT.values().to(self.device)
        self.event_driven = True

        # exact integration of the linear (v, g) system over one step
        dt = params.dt
        self.a_m = math.exp(-dt / params.t_mbr)
        self.a_s = math.exp(-dt / params.tau)
        self.k = params.tau / (params.tau - params.t_mbr)  # g -> v coupling

        self.delay_steps = max(1, round(params.t_dly / dt))
        self.rfc_steps = round(params.t_rfc / dt)
        self.poisson_w = params.w_syn * params.f_poi

        # external drive: Poisson rate (Hz) per neuron; silenced mask
        self.rate = torch.zeros(n, batch, device=self.device)
        self._driven = None
        self.silenced = torch.zeros(n, batch, device=self.device, dtype=torch.bool)
        self.reset()

    # ---- state -------------------------------------------------------------
    def reset(self):
        n, b, d = self.n, self.batch, self.device
        self.v = torch.full((n, b), self.p.v_0, device=d)
        self.g = torch.zeros(n, b, device=d)
        self.rfc = torch.zeros(n, b, device=d, dtype=torch.int32)  # steps left
        # ring buffer of spikes waiting out the synaptic delay
        self.buf = torch.zeros(self.delay_steps, n, b, device=d)
        self.t_step = 0

    def ids(self, flyids) -> torch.Tensor:
        """FlyWire root IDs -> model indices (IDs missing from v783 are skipped)."""
        idx = [self.flyid2i[f] for f in flyids if f in self.flyid2i]
        return torch.as_tensor(idx, dtype=torch.int64, device=self.device)

    def set_rate(self, idx: torch.Tensor, hz, batch_slice=slice(None)):
        """Drive neurons with Poisson input (like optogenetic activation)."""
        self.rate[idx, batch_slice] = torch.as_tensor(hz, dtype=torch.float32, device=self.device)
        self._update_driven()

    def clear_rates(self):
        self.rate.zero_()
        self._update_driven()

    def _update_driven(self):
        driven = torch.nonzero(self.rate.any(dim=1)).flatten()
        self._driven = driven if len(driven) else None

    def silence(self, idx: torch.Tensor, batch_slice=slice(None)):
        """Cut all output synapses of these neurons (like the paper's silencing)."""
        self.silenced[idx, batch_slice] = True

    # ---- simulation --------------------------------------------------------
    @torch.no_grad()
    def step(self) -> torch.Tensor:
        """Advance one dt. Returns the spike matrix (n, batch) as float."""
        p = self.p
        # spikes emitted `delay_steps` ago arrive now
        slot = self.t_step % self.delay_steps
        arriving = self.buf[slot]
        if self.event_driven:
            self._propagate(arriving)
        else:
            self.g += torch.sparse.mm(self.W, arriving) if self.batch > 1 else \
                torch.mv(self.W, arriving[:, 0]).unsqueeze(1)

        # integrate neurons that are not refractory
        active = self.rfc <= 0
        u = self.v - p.v_0
        u_new = self.k * self.g * (self.a_s - self.a_m) + u * self.a_m
        self.v = torch.where(active, p.v_0 + u_new, self.v)
        self.g = torch.where(active, self.g * self.a_s, self.g)

        # external Poisson input kicks v directly (only on driven neurons)
        poisson_target = None
        if self._driven is not None:
            prob = self.rate[self._driven] * (p.dt * 1e-3)
            kick = torch.rand(prob.shape, device=self.device, generator=self.gen) < prob
            self.v[self._driven] += kick * self.poisson_w
            poisson_target = self._driven

        spiked = self.v > p.v_th
        self.v = torch.where(spiked, torch.full_like(self.v, p.v_rst), self.v)
        self.g = torch.where(spiked, torch.zeros_like(self.g), self.g)
        self.rfc = torch.where(spiked, self.rfc_steps, self.rfc - 1).to(torch.int32)
        if poisson_target is not None:  # driven neurons have no refractory period
            self.rfc[poisson_target] = 0

        out = spiked & ~self.silenced
        self.buf[slot] = out.float()
        self.t_step += 1
        return out

    def _propagate(self, arriving: torch.Tensor):
        """g[post, b] += w for every synapse of every neuron that spiked in column b."""
        ev = torch.nonzero(arriving)  # (k, 2): presynaptic index, batch column
        if len(ev) == 0:
            return
        pre, col = ev[:, 0], ev[:, 1]
        start = self.out_ptr[pre]
        cnt = self.out_ptr[pre + 1] - start
        csum = torch.cumsum(cnt, 0)
        total = int(csum[-1])
        which = torch.repeat_interleave(torch.arange(len(ev), device=self.device), cnt, output_size=total)
        pos = start[which] + torch.arange(total, device=self.device) - (csum - cnt)[which]
        flat = self.out_post[pos] * self.batch + col[which]
        self.g.view(-1).index_add_(0, flat, self.out_w[pos])

    @torch.no_grad()
    def run(self, ms: float, record: torch.Tensor | None = None) -> torch.Tensor:
        """Run for `ms` milliseconds; return spike counts (n or len(record), batch)."""
        steps = round(ms / self.p.dt)
        counts = torch.zeros(self.n if record is None else len(record), self.batch, device=self.device)
        for _ in range(steps):
            s = self.step()
            counts += (s if record is None else s[record]).float()
        return counts
