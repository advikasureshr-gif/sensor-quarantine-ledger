"""The blockchain layer: blocks verify, naive edits are caught by one node, skilled rewrites only by the majority."""
from iotguard.chain import Blockchain, Network, merkle_root, relink
from iotguard.security import Ledger


def make_records(n=10):
    led = Ledger(clock=lambda: "2026-10-22T10:00:00")
    for i in range(n):
        led.append("alert" if i % 3 == 0 else "baseline_checkpoint", float(i), k=i, sensor="pir")
    return led.records


def test_build_and_verify():
    chain = Blockchain.build(make_records(10), block_size=4, difficulty=2)
    assert len(chain.blocks) == 3 and len(chain.blocks[-1].records) == 2
    assert chain.verify() == (True, None, "intact")
    assert all(b.hash.startswith("00") for b in chain.blocks)


def test_merkle_root_changes_with_any_leaf_and_handles_odd_counts():
    a = ["a" * 64, "b" * 64, "c" * 64]
    assert merkle_root(a) != merkle_root(["a" * 64, "b" * 64, "d" * 64])
    assert merkle_root(a) == merkle_root(a)
    assert merkle_root([]) != merkle_root(a)


def test_single_edit_is_caught_by_the_chain_itself():
    net = Network(Blockchain.build(make_records(), 4, 2), 3)
    net.edit_record("node-B", 5, lambda r: r.update(event="sensor_recovered"))
    ok, block, why = net.node("node-B").chain.verify()
    assert not ok and block == 1 and "#5" in why
    assert net.node("node-A").chain.verify()[0]


def test_skilled_rewrite_fools_one_node_but_not_the_vote():
    net = Network(Blockchain.build(make_records(), 4, 2), 3)
    net.rewrite_history("node-C", 5, lambda r: r["data"].update(sensor="ldr"))
    assert net.node("node-C").chain.verify()[0], "a consistent rewrite passes a lone node's own checks"
    v = net.vote()
    assert v.quorum and list(v.dissenters) == ["node-C"] and v.dissenters["node-C"] == 1
    assert v.agreed_blocks == v.total_blocks
    assert net.heal() == ["node-C"] and net.vote().clean


def test_no_quorum_when_every_node_disagrees():
    net = Network(Blockchain.build(make_records(), 4, 2), 3)
    for name in ("node-A", "node-B", "node-C"):
        net.rewrite_history(name, 5, lambda r, n=name: r["data"].update(who=n))
    v = net.vote()
    assert not v.quorum and net.heal() == []


def test_relink_matches_a_real_ledger():
    recs = make_records(6)
    assert relink(recs) == recs
