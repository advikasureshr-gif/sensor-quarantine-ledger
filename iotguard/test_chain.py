"""
chain.py - a small permissioned blockchain over the event ledger.

    python -m iotguard.chain logs/run_ledger.jsonl            build, verify, replicate to 3 nodes, attack, vote

What it adds to the plain hash-chained ledger (security.Ledger):
  * BLOCKS: ledger records are batched; each block header holds the previous block's hash, a MERKLE ROOT of its records,
    a timestamp and a nonce found by a (deliberately tiny) PROOF OF WORK.
  * REPLICAS: several nodes each hold a full copy and VOTE per block (simple majority). A lone hash chain cannot detect an
    attacker who rewrites a whole file and recomputes every hash; a majority of independent copies can.

Honest scope: the nodes here run in one process (a simulation of replicas, not a network), the proof of work is only
there to make rewriting history cost something, and membership is fixed (permissioned). It reads ledger records; it never
changes what the detector decides.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

from .security import Ledger

DEFAULT_DIFFICULTY = 3            # leading hex zeros required (about 4,000 hashes per block, well under a second)
DEFAULT_BLOCK_SIZE = 4            # ledger records per block


def _h(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def merkle_root(leaves: list[str]) -> str:
    """Root of a binary hash tree over the given hex hashes (the last one is duplicated when a level is odd)."""
    if not leaves:
        return _h("")
    level = list(leaves)
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [_h(level[i] + level[i + 1]) for i in range(0, len(level), 2)]
    return level[0]


def relink(records: list[dict]) -> list[dict]:
    """Recompute seq / prev / hash for a list of ledger records (what an attacker who edits one must do)."""
    out, prev = [], Ledger.GENESIS
    for i, r in enumerate(records):
        core = {k: v for k, v in r.items() if k != "hash"}
        core["seq"], core["prev"] = i, prev
        rec = {**core, "hash": Ledger._hash(prev, core)}
        out.append(rec)
        prev = rec["hash"]
    return out


@dataclass
class Block:
    index: int
    prev_hash: str
    timestamp: str
    merkle: str
    difficulty: int
    nonce: int
    records: list[dict]
    hash: str = ""

    def header(self, nonce: Optional[int] = None) -> str:
        n = self.nonce if nonce is None else nonce
        return f"{self.index}|{self.prev_hash}|{self.timestamp}|{self.merkle}|{self.difficulty}|{n}"

    def compute_hash(self) -> str:
        return _h(self.header())

    def mine(self) -> "Block":
        target, n = "0" * self.difficulty, 0
        while not _h(self.header(n)).startswith(target):
            n += 1
        self.nonce, self.hash = n, _h(self.header(n))
        return self


class Blockchain:
    GENESIS = "0" * 64

    def __init__(self, blocks: Optional[list[Block]] = None) -> None:
        self.blocks: list[Block] = blocks or []

    @classmethod
    def build(cls, records: list[dict], block_size: int = DEFAULT_BLOCK_SIZE, difficulty: int = DEFAULT_DIFFICULTY) -> "Blockchain":
        chain, prev = cls(), cls.GENESIS
        for i in range(0, len(records), block_size):
            part = copy.deepcopy(records[i:i + block_size])
            b = Block(index=len(chain.blocks), prev_hash=prev, timestamp=str(part[-1].get("wall", "")),
                      merkle=merkle_root([r["hash"] for r in part]), difficulty=difficulty, nonce=0, records=part).mine()
            chain.blocks.append(b)
            prev = b.hash
        return chain

    @property
    def head(self) -> str:
        return self.blocks[-1].hash if self.blocks else self.GENESIS

    def all_records(self) -> list[dict]:
        return [r for b in self.blocks for r in b.records]

    def verify(self) -> tuple[bool, Optional[int], str]:
        """(ok, index of the first bad block, why). Checks links, proof of work, Merkle roots and the ledger hash chain."""
        prev = self.GENESIS
        for b in self.blocks:
            if b.prev_hash != prev:
                return False, b.index, "block does not link to the previous block"
            if b.compute_hash() != b.hash or not b.hash.startswith("0" * b.difficulty):
                return False, b.index, "block hash or proof of work is wrong"
            if merkle_root([r["hash"] for r in b.records]) != b.merkle:
                return False, b.index, "records do not match the block's Merkle root"
            prev = b.hash
        ok, bad = _ledger_ok(self.all_records())
        if not ok:
            seen = 0
            for b in self.blocks:
                seen += len(b.records)
                if bad < seen:
                    return False, b.index, f"ledger record #{bad} does not match its hash"
        return True, None, "intact"


def _ledger_ok(records: list[dict]) -> tuple[bool, Optional[int]]:
    led = Ledger()
    led.records = records
    return led.verify()


# ---------------------------------------------------------------------------
# replicas and voting
# ---------------------------------------------------------------------------
@dataclass
class Node:
    name: str
    chain: Blockchain


@dataclass
class Verdict:
    agreed_blocks: int
    total_blocks: int
    dissenters: dict = field(default_factory=dict)       # node name -> first block index where it disagrees with the majority
    quorum: bool = True

    @property
    def clean(self) -> bool:
        return self.quorum and not self.dissenters


class Network:
    """`n` replicas of one chain. Each block is accepted only if a strict majority of nodes hold the same block hash."""

    def __init__(self, chain: Blockchain, n: int = 3, names: Optional[list[str]] = None) -> None:
        names = names or [f"node-{chr(65 + i)}" for i in range(n)]
        self.nodes = [Node(nm, copy.deepcopy(chain)) for nm in names]

    def node(self, name: str) -> Node:
        return next(n for n in self.nodes if n.name == name)

    # --- attacks on one replica ----------------------------------------------------------------
    def edit_record(self, name: str, seq: int, mutate: Callable[[dict], None]) -> None:
        """Naive attack: change one stored record and nothing else."""
        for b in self.node(name).chain.blocks:
            for r in b.records:
                if r["seq"] == seq:
                    mutate(r)
                    return
        raise KeyError(seq)

    def rewrite_history(self, name: str, seq: int, mutate: Callable[[dict], None]) -> None:
        """Skilled attack: change a record, then recompute every ledger hash, Merkle root and block (re-mining them)."""
        node = self.node(name)
        recs = copy.deepcopy(node.chain.all_records())
        mutate(recs[seq])
        size = len(node.chain.blocks[0].records)
        diff = node.chain.blocks[0].difficulty
        node.chain = Blockchain.build(relink(recs), block_size=size, difficulty=diff)

    # --- voting ---------------------------------------------------------------------------------
    def vote(self) -> Verdict:
        n = len(self.nodes)
        top = max(len(nd.chain.blocks) for nd in self.nodes)
        v = Verdict(agreed_blocks=0, total_blocks=top)
        majority: list[Optional[str]] = []
        for i in range(top):
            votes = Counter(nd.chain.blocks[i].hash for nd in self.nodes if i < len(nd.chain.blocks))
            h, c = votes.most_common(1)[0]
            if c * 2 > n:
                majority.append(h)
                v.agreed_blocks += 1
            else:
                majority.append(None)
                v.quorum = False
        for nd in self.nodes:
            for i, h in enumerate(majority):
                mine = nd.chain.blocks[i].hash if i < len(nd.chain.blocks) else None
                if h is not None and mine != h:
                    v.dissenters[nd.name] = i
                    break
        return v

    def heal(self) -> list[str]:
        """Replace each dissenting replica by a copy of an agreeing one. Returns the names that were repaired."""
        v = self.vote()
        if not v.quorum or not v.dissenters:
            return []
        good = next(nd for nd in self.nodes if nd.name not in v.dissenters)
        for nd in self.nodes:
            if nd.name in v.dissenters:
                nd.chain = copy.deepcopy(good.chain)
        return list(v.dissenters)


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------
def _load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _main(argv=None) -> int:
    path = (argv or sys.argv[1:] or [None])[0]
    if not path:
        print(__doc__)
        return 2
    records = _load(path)
    ok, bad = _ledger_ok(records)
    if not ok:
        print(f"the ledger itself is already broken at record #{bad}")
        return 1
    chain = Blockchain.build(records)
    print(f"{len(records)} ledger records -> {len(chain.blocks)} blocks (difficulty {chain.blocks[0].difficulty}, {DEFAULT_BLOCK_SIZE} records each)")
    for b in chain.blocks:
        print(f"  block {b.index}: {len(b.records)} records  merkle {b.merkle[:10]}..  nonce {b.nonce:<6} hash {b.hash[:12]}..")
    print("chain verify:", chain.verify()[2])
    return 0


if __name__ == "__main__":
    sys.exit(_main())
