"""B2, B3, B6 fixture tests: the Lightning ballot and the Nemotron planner.

No network and no real models: `vlm.chat` is monkeypatched for behaviour and
`vlm._post` is monkeypatched for the one test that has to inspect the actual wire
payload. What is under test is the arithmetic a judge checks with a calculator
(doubt equals 1 minus agreement, floored at 0.05), the structured-output syntax
that actually constrains on this vLLM build (structured_outputs.choice yes,
guided_json never), the exact section 7 Agency plan shape, and the self-recovery
beat firing exactly once and then falling back with an honest drafted_by.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import ballot, config, contracts, db, planner, scorer, vlm  # noqa: E402

AOI = [-82.78, 27.75, -82.70, 27.82]
CENTROID = [-82.74, 27.78]


# ------------------------------------------------------------------- fixtures
@pytest.fixture
def store(tmp_path, monkeypatch):
    """A private database plus a clean thread-local connection and clean stats."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "AOI", list(AOI))
    monkeypatch.setattr(db, "_local", threading.local())
    db.init()
    ballot.reset_stats()
    planner.reset_stats()
    vlm.reset_breakers()
    yield tmp_path
    ballot.reset_stats()
    planner.reset_stats()


def add_building(
    fid: str,
    *,
    cls: int = 3,
    conf: float = 0.8,
    label: str = "",
    centroid=None,
    facility=None,
    svi: float = 0.5,
    last_seen_at: float = 1.0,
    confirmed: int = 0,
) -> None:
    db.run(
        """INSERT INTO buildings
             (footprint_id, label, centroid_json, geom_json, damage_class, confidence,
              graded_by, confirmed, facility_json, svi, area_m2, last_seen_at, source_tile)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            fid,
            label or fid,
            json.dumps(centroid or CENTROID),
            None,
            int(cls),
            float(conf),
            "nemotron-vl",
            int(confirmed),
            json.dumps(facility) if facility else None,
            float(svi),
            180.0,
            float(last_seen_at),
            "tile.jpg",
        ),
    )


class FakeBuilding:
    """The ingest-path shape: a grading.GradedBuilding carries its own caption."""

    def __init__(self, fid, cls=3, conf=0.8, caption="", label="", area_m2=180.0, facility=None):
        self.footprint_id = fid
        self.cls = cls
        self.conf = conf
        self.caption = caption
        self.label = label or fid
        self.area_m2 = area_m2
        self.centroid = list(CENTROID)
        self.facility_near = facility
        self.svi = 0.5
        self.graded_by = "nemotron-vl"


class Recorder:
    """A scripted vlm.chat. Thread safe, because the ballot samples in parallel.

    `scripted` keys replies by a substring of the prompt rather than by global call
    order. A corpus sweep interleaves k samples for every building across the pool,
    so order-keyed replies hand building A's votes to building B depending on
    thread scheduling, and the test then passes or fails by luck. Keying on
    something in the prompt (a distinct caption) makes the assertion about the
    ballot rather than about the scheduler.
    """

    def __init__(self, replies=(), how=vlm.GRADE_HOW_MODEL, scripted=None):
        self._replies = list(replies)
        # Longest key first: "caption for b1" is a prefix of "caption for b11", and
        # first-match-wins would let b11 drain b1's queue and silently stub a row.
        self._scripted = [
            (k, list(v))
            for k, v in sorted((scripted or {}).items(), key=lambda kv: -len(kv[0]))
        ]
        self._how = how
        self._lock = threading.Lock()
        self.calls: list[dict] = []

    def __call__(self, endpoint, messages, **kw):
        prompt = messages[-1]["content"]
        with self._lock:
            self.calls.append({"endpoint": endpoint, "messages": list(messages), "kw": dict(kw)})
            queue = self._replies
            for key, scripted in self._scripted:
                if key in prompt:
                    queue = scripted
                    break
            reply = queue.pop(0) if queue else None
        if reply is None:
            return vlm._fallback_text(kw.get("schema"), kw.get("choice")), vlm.GRADE_HOW_STUB
        if isinstance(reply, tuple):
            return reply
        return reply, self._how

    @property
    def prompts(self) -> list[str]:
        return [c["messages"][-1]["content"] for c in self.calls]


def install(monkeypatch, recorder: Recorder) -> Recorder:
    monkeypatch.setattr(vlm, "chat", recorder)
    return recorder


# --------------------------------------------------------------- the arithmetic
def test_votes_are_sampled_labels_in_order_not_tallies(store, monkeypatch):
    """The console renders "AI checked 8x: 6x destroyed, 2x major" from these."""
    sampled = ["3", "3", "2", "3", "3", "2", "3", "3"]
    install(monkeypatch, Recorder(sampled))

    result = ballot.vote(FakeBuilding("b1", caption="roof collapsed"), k=8)

    assert len(result.votes) == 8, "one entry per sample, not one entry per class"
    assert sorted(result.votes) == sorted(int(s) for s in sampled)
    assert result.votes.count(3) == 6 and result.votes.count(2) == 2
    # A tally would be a 4-entry histogram. This is not that.
    assert set(result.votes) <= {0, 1, 2, 3}
    assert result.voted_class == 3
    assert result.how == vlm.GRADE_HOW_MODEL


def test_unanimous_ballot_lands_exactly_on_the_floor(store, monkeypatch):
    install(monkeypatch, Recorder(["3"] * 8))

    result = ballot.vote(FakeBuilding("b1", caption="destroyed"), k=8)

    assert result.vote_agreement == 1.0
    assert result.doubt == contracts.DOUBT_FLOOR
    assert result.at_floor is True


def test_five_three_split_is_doubt_zero_point_three_seven_five(store, monkeypatch):
    install(monkeypatch, Recorder(["3", "3", "2", "3", "2", "3", "2", "3"]))

    result = ballot.vote(FakeBuilding("b1", caption="partial collapse"), k=8)

    assert result.vote_agreement == 0.625
    assert result.doubt == 0.375
    assert result.voted_class == 3


@pytest.mark.parametrize(
    "votes, agreement, doubt",
    [
        (["3"] * 8, 1.0, 0.05),
        (["3"] * 7 + ["2"], 0.875, 0.125),
        (["3"] * 6 + ["2", "2"], 0.75, 0.25),
        (["3"] * 5 + ["2", "2", "2"], 0.625, 0.375),
        (["3", "3", "2", "2", "1", "1", "0", "0"], 0.25, 0.75),
    ],
)
def test_doubt_is_one_minus_agreement_floored(store, monkeypatch, votes, agreement, doubt):
    install(monkeypatch, Recorder(votes))
    result = ballot.vote(FakeBuilding("b1", caption="damage"), k=8)
    assert result.vote_agreement == agreement
    assert result.doubt == doubt
    assert result.doubt >= contracts.DOUBT_FLOOR


def test_a_tie_takes_the_higher_severity(store, monkeypatch):
    """Rounding damage down on a split vote is the one bias a triage tool cannot have."""
    install(monkeypatch, Recorder(["3", "3", "2", "2"]))
    result = ballot.vote(FakeBuilding("b1", caption="damage"), k=4)
    assert result.voted_class == 3
    assert result.vote_agreement == 0.5


# ------------------------------------------------------------ the caption input
def test_the_caption_reaches_lightning(store, monkeypatch):
    """The caption IS the ballot. Lightning never sees pixels, so cross-examining
    the grade against the caption is the entire mechanism."""
    rec = install(monkeypatch, Recorder(["3"] * 8))
    caption = "two-storey wood structure, roof collapsed, standing water in the street"

    ballot.vote(
        FakeBuilding("b1", cls=1, conf=0.44, caption=caption, area_m2=210.0), k=8
    )

    assert rec.calls, "the ballot must actually call the model"
    for prompt in rec.prompts:
        assert caption in prompt, "the free-text caption must be in the prompt verbatim"
        assert "minor damage" in prompt, "the structured grade must be there to contradict it"
        assert "0.44" in prompt, "the grader confidence is part of the ballot input"
        assert "210 m2" in prompt, "join context: footprint area"


def test_join_context_reaches_lightning(store, monkeypatch):
    rec = install(monkeypatch, Recorder(["2"] * 8))
    add_building("neighbour", cls=3, centroid=[CENTROID[0] + 0.0002, CENTROID[1]])
    ballot.reset_stats()  # drop the neighbour cache so the new row is visible

    ballot.vote(
        FakeBuilding(
            "b1",
            caption="damaged roof",
            facility={"name": "Providence Mount", "type": "nursing_home", "dist_m": 80},
        ),
        k=4,
    )

    prompt = rec.prompts[0]
    assert "Providence Mount" in prompt and "nursing_home" in prompt and "80 m" in prompt
    assert "Neighbouring structures graded" in prompt


def test_every_ballot_call_goes_to_lightning(store, monkeypatch):
    rec = install(monkeypatch, Recorder(["3"] * 8))
    ballot.vote(FakeBuilding("b1", caption="c"), k=8)
    assert {c["endpoint"].name for c in rec.calls} == {"lightning"}


def test_a_missing_caption_is_named_not_invented(store, monkeypatch):
    """A withheld tile leaves no caption behind. The prompt says so."""
    rec = install(monkeypatch, Recorder(["3"] * 4))
    ballot.vote(FakeBuilding("b1", caption=""), k=4)
    assert "caption: none available" in rec.prompts[0]


# ---------------------------------------------------- the verified wire syntax
def test_the_ballot_sends_a_constraint_that_actually_constrains(store, monkeypatch):
    """On this vLLM build (0.25.1) BOTH guided_json and guided_choice are ignored.

    Measured on the box: with guided_choice alone Lightning returned "Here's a
    thinking" and finish_reason "length"; with structured_outputs it returned "3"
    and finish_reason "stop". So the enumerated pick MUST carry
    structured_outputs.choice, and guided_json must never appear anywhere.

    Asserted at the transport layer, so the real payload builder runs: a test
    against a patched chat() would prove nothing about the wire.
    """
    payloads: list[dict] = []
    lock = threading.Lock()

    def fake_post(url, payload, timeout):
        with lock:
            payloads.append(json.loads(json.dumps(payload)))
        picks = (payload.get("structured_outputs") or {}).get("choice")
        if picks:
            return {"choices": [{"message": {"content": picks[0]}}]}
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"captions": [{"index": 0, "tags": ["collapsed roof"]}]}
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(vlm, "_post", fake_post)

    ballot.vote(FakeBuilding("b1", caption="roof collapsed"), k=8)
    ballot.extract_tags(["roof collapsed, standing water"])

    assert len(payloads) == 9, "eight ballot samples plus one tag sweep"
    for p in payloads:
        assert "guided_json" not in p, "guided_json is silently ignored on this build"

    ballots = [p for p in payloads if "structured_outputs" in p]
    assert len(ballots) == 8, "every ballot sample must carry the working constraint"
    for p in ballots:
        assert p["structured_outputs"]["choice"] == ["0", "1", "2", "3"]
        assert "response_format" not in p, "an enumerated pick is a choice, not a schema"
        assert p["temperature"] == ballot.BALLOT_TEMPERATURE

    tags = [p for p in payloads if "response_format" in p]
    assert len(tags) == 1
    rf = tags[0]["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["schema"] == ballot.TAGS_SCHEMA


def test_plan_and_flight_use_response_format_json_schema(store, monkeypatch):
    payloads: list[dict] = []

    def fake_post(url, payload, timeout):
        payloads.append(json.loads(json.dumps(payload)))
        return {"choices": [{"message": {"content": "{}"}}]}

    monkeypatch.setattr(vlm, "_post", fake_post)
    add_building("b1")
    items = scorer.rank(limit=5)["items"]

    planner.draft_plan(items, {"fire": 3}, force_invalid_first=False)
    planner.next_flight(items, [], force_invalid_first=False)

    assert payloads, "the planner must actually call Nano"
    schemas = set()
    for p in payloads:
        assert "guided_json" not in p
        assert p["response_format"]["type"] == "json_schema"
        schemas.add(p["response_format"]["json_schema"]["name"])
        # max_tokens IS the latency budget at 24 tok/s, so it stays sized.
        assert p["max_tokens"] <= planner.FLIGHT_MAX_TOKENS
    assert schemas == {"agency_plan", "flight_plan"}


def test_the_plan_schema_pins_one_entry_per_building(store, monkeypatch):
    """minItems == maxItems == the building count, so the decoder cannot truncate.

    The triple is constrained POSITIONALLY because of a measured failure: with a
    bare integer type in all three slots Nano wrote units 0 (meaning "no units
    needed") on 1 of 10 first attempts, and each re-prompt doubles the beat at 24
    tok/s. Verified on the box: with prefixItems, 6 of 6 first attempts valid.
    """
    schema = planner.plan_schema(7)
    assert schema["properties"]["a"]["minItems"] == 7
    assert schema["properties"]["a"]["maxItems"] == 7
    item = schema["properties"]["a"]["items"]
    assert item["minItems"] == 3 and item["maxItems"] == 3
    agency, task, units = item["prefixItems"]
    assert agency["maximum"] == len(contracts.AGENCIES) - 1
    assert task["maximum"] == len(planner.TASK_VOCAB) - 1
    assert units["minimum"] == 1, "zero units is TASK_NO_ACTION, never a unit count"
    assert units["maximum"] == planner.MAX_UNITS
    assert item["items"] is False, "the tail is closed so a fourth element cannot appear"


def test_no_think_on_cheap_calls_and_reasoning_on_for_the_replan(store, monkeypatch):
    rec = install(monkeypatch, Recorder([], how=vlm.GRADE_HOW_STUB))
    add_building("b1")
    items = scorer.rank(limit=5)["items"]

    planner.draft_plan(items, {"fire": 3}, force_invalid_first=False)
    plan_prompts = list(rec.prompts)
    assert all("/no_think" in p for p in plan_prompts), "the plan draft is a cheap structured call"

    rec.calls.clear()
    planner.next_flight(items, [], force_invalid_first=False)
    assert rec.prompts, "the flight call must happen"
    assert all("/no_think" not in p for p in rec.prompts), "reasoning is ON for the replan beat"


# ------------------------------------------------------------- the stub honesty
def test_a_stub_ballot_reports_no_tally_but_still_sets_doubt(store, monkeypatch):
    """A sample that never came back from the model is not a vote."""
    install(monkeypatch, Recorder([], how=vlm.GRADE_HOW_STUB))

    result = ballot.vote(FakeBuilding("b1", conf=0.6, caption="c"), k=8)

    assert result.how == vlm.GRADE_HOW_STUB
    assert result.votes == []
    assert result.vote_agreement is None
    assert result.doubt == pytest.approx(0.4), "1 minus grader confidence, per section 7"
    assert result.wire()["votes"] is None, "null reads as 'no ballot yet' in the console"


def test_too_few_returned_samples_reports_the_stub_path(store, monkeypatch):
    """Two samples can only ever say 0.5 or 1.0. That is not a measured agreement."""
    install(monkeypatch, Recorder(["3", "3"]))

    result = ballot.vote(FakeBuilding("b1", conf=0.7, caption="c"), k=8)

    assert result.how == vlm.GRADE_HOW_STUB
    assert result.votes == []
    assert result.doubt == pytest.approx(0.3)


def test_persist_writes_the_ballot_and_a_stub_writes_nulls(store, monkeypatch):
    add_building("b1", conf=0.7)
    install(monkeypatch, Recorder(["3"] * 6 + ["2", "2"]))
    ballot.persist("b1", ballot.vote(FakeBuilding("b1", caption="c"), k=8))

    row = db.q1("SELECT doubt, votes_json, vote_agreement FROM buildings WHERE footprint_id='b1'")
    assert row["doubt"] == 0.25
    assert row["vote_agreement"] == 0.75
    assert json.loads(row["votes_json"]) == [3, 3, 3, 3, 3, 3, 2, 2]

    install(monkeypatch, Recorder([], how=vlm.GRADE_HOW_STUB))
    ballot.persist("b1", ballot.vote(FakeBuilding("b1", conf=0.7, caption="c"), k=8))
    row = db.q1("SELECT doubt, votes_json, vote_agreement FROM buildings WHERE footprint_id='b1'")
    assert row["votes_json"] is None and row["vote_agreement"] is None
    assert row["doubt"] == pytest.approx(0.3)


def test_the_ballot_feeds_the_rank(store, monkeypatch):
    """doubt is a multiplier in priority, so a contested building must RISE."""
    add_building("agreed", cls=3, conf=0.9, last_seen_at=1.0)
    add_building("contested", cls=3, conf=0.9, last_seen_at=1.0)
    install(monkeypatch, Recorder(["3"] * 8))
    ballot.persist("agreed", ballot.vote(FakeBuilding("agreed", caption="c"), k=8))
    install(monkeypatch, Recorder(["3", "3", "3", "2", "2", "1", "1", "0"]))
    ballot.persist("contested", ballot.vote(FakeBuilding("contested", caption="c"), k=8))

    items = scorer.rank(limit=10)["items"]
    order = [it["footprint_id"] for it in items]
    assert order[0] == "contested", "uncertainty means send someone to look"
    doubts = {it["footprint_id"]: it["inputs"]["doubt"] for it in items}
    assert doubts["agreed"] == contracts.DOUBT_FLOOR
    assert doubts["contested"] > contracts.DOUBT_FLOOR


# ------------------------------------------------------------ the corpus sweep
def test_vote_batch_votes_every_building_and_persists(store, monkeypatch):
    install(monkeypatch, Recorder(["2"] * 400))
    buildings = [FakeBuilding(f"b{i}", cls=2, caption="damaged wall") for i in range(50)]

    results = ballot.vote_batch(buildings, k=8, max_concurrency=16)

    assert len(results) == 50
    assert all(len(r.votes) == 8 for r in results)
    assert {r.footprint_id for r in results} == {b.footprint_id for b in buildings}
    sweep = ballot.last_sweep()
    assert sweep["buildings"] == 50 and sweep["k"] == 8 and sweep["selection"] == "all"


def test_an_exhausted_budget_degrades_to_labelled_stubs(store, monkeypatch):
    """A slow endpoint must never stall the caller. It must go stub instead."""
    install(monkeypatch, Recorder(["3"] * 400))
    buildings = [FakeBuilding(f"b{i}", conf=0.6, caption="c") for i in range(4)]

    results = ballot.vote_batch(buildings, k=8, budget_s=-1.0)

    assert len(results) == 4
    assert all(r.how == vlm.GRADE_HOW_STUB for r in results)
    assert all(r.votes == [] for r in results)
    assert all(r.doubt == pytest.approx(0.4) for r in results)


def test_the_budget_bounds_the_sweep_not_each_call(store, monkeypatch):
    """The queued tail must go stub the moment the deadline passes.

    A measured bug: the timeout used to be computed where the task was QUEUED, so
    with 16 workers and 96 samples each call got a fresh full timeout and the sweep
    overran. On the box a 12-building tile spent 5623 ms against a 4 s budget. The
    deadline is now read at execution, so a sweep whose budget expires mid-flight
    stops opening sockets instead of finishing at its own pace.
    """
    calls: list[float] = []
    lock = threading.Lock()
    budget_s = 0.35

    def slow(endpoint, messages, **kw):
        with lock:
            calls.append(time.monotonic())
        time.sleep(0.05)
        return "3", vlm.GRADE_HOW_MODEL

    monkeypatch.setattr(vlm, "chat", slow)
    buildings = [FakeBuilding(f"b{i}", conf=0.6, caption=f"cap {i}") for i in range(40)]

    t0 = time.monotonic()
    results = ballot.vote_batch(buildings, k=8, max_concurrency=4, budget_s=budget_s)
    elapsed = time.monotonic() - t0

    assert len(results) == 40, "every building still gets a row"
    assert elapsed < budget_s + 2.0, f"the sweep ran {elapsed:.2f}s against a {budget_s}s budget"
    assert len(calls) < 40 * 8, "the tail must not have opened a socket at all"
    assert any(r.how == vlm.GRADE_HOW_STUB for r in results), "the tail degraded"
    assert ballot.last_sweep()["budget_hit"] is True
    for r in results:
        if r.how == vlm.GRADE_HOW_STUB:
            assert r.votes == [] and r.doubt == pytest.approx(0.4)


def test_a_sample_with_too_little_budget_left_never_opens_a_socket(store, monkeypatch):
    """A cutoff we imposed must not be charged to Lightning.

    Measured on the box: a tile whose budget expired mid-sweep tripped vlm's
    circuit breaker, and the NEXT tile then got instant stubs for the whole 30 s
    cooldown from a Lightning that was answering fine. The first guard is that a
    sample without MIN_SAMPLE_S left does not call at all, so there is no cut-off
    request for the breaker to count.
    """
    calls: list[int] = []
    monkeypatch.setattr(
        vlm, "chat", lambda e, m, **kw: (calls.append(1), ("3", vlm.GRADE_HOW_MODEL))[1]
    )

    # Just under the per-sample floor: enough clock left to look non-zero, not
    # enough to be worth a round trip.
    results = ballot.vote_batch(
        [FakeBuilding("b1", conf=0.6, caption="c")],
        k=8,
        budget_s=ballot.MIN_SAMPLE_S * 0.5,
    )

    assert calls == [], "no socket opened when the remaining budget is below the floor"
    assert results[0].how == vlm.GRADE_HOW_STUB
    assert results[0].doubt == pytest.approx(0.4)
    assert not vlm._breaker_open("lightning"), "our own cutoff must not open the breaker"


def test_a_partly_served_sweep_clears_the_breaker(store, monkeypatch):
    """The second guard, asserted on the mechanism rather than on a stopwatch.

    A sweep that hit its budget while the endpoint was demonstrably answering
    clears any breaker state the cutoff contributed, so the next tile is not served
    instant stubs by a healthy Lightning.
    """
    install(monkeypatch, Recorder(["3"] * 8))  # exactly one building's worth

    # 8 replies for 2 buildings: the first votes, the second falls through to stub.
    results = ballot.vote_batch(
        [FakeBuilding("first", conf=0.6, caption="a"), FakeBuilding("second", conf=0.6, caption="b")],
        k=8,
        budget_s=30.0,
    )

    by_id = {r.footprint_id: r for r in results}
    assert by_id["first"].how == vlm.GRADE_HOW_MODEL
    assert by_id["second"].how == vlm.GRADE_HOW_STUB

    # Now the same mixed outcome under a budget, which is the state that used to
    # leave a poisoned breaker behind.
    for _ in range(vlm.BREAKER_TRIP + 1):
        vlm._breaker_note("lightning", ok=False)
    assert vlm._breaker_open("lightning"), "precondition: the breaker is open"

    install(monkeypatch, Recorder(["3"] * 8))
    ballot.vote_batch(
        [FakeBuilding("third", conf=0.6, caption="c"), FakeBuilding("fourth", conf=0.6, caption="d")],
        k=8,
        budget_s=30.0,
    )
    assert ballot.last_sweep()["budget_hit"] is True
    assert not vlm._breaker_open("lightning"), "a partly-served sweep heals the breaker"


def test_uncertain_only_spends_a_tight_budget_on_the_least_certain(store, monkeypatch):
    install(monkeypatch, Recorder(["3"] * 400))
    buildings = [
        FakeBuilding("sure", cls=3, conf=0.95, caption="building destroyed"),
        FakeBuilding("unsure", cls=3, conf=0.30, caption="building destroyed"),
        FakeBuilding("contradicted", cls=0, conf=0.90, caption="roof collapsed, walls down"),
    ]

    results = ballot.vote_batch(buildings, k=4, uncertain_only=2)

    voted = [r.footprint_id for r in results]
    assert len(voted) == 2
    assert "contradicted" in voted, "a caption contradicting the grade goes first"
    assert "unsure" in voted
    assert "sure" not in voted
    assert "uncertain-only" in ballot.last_sweep()["selection"]


def test_uncertain_first_orders_contradiction_then_confidence(store):
    ordered = ballot.uncertain_first(
        [
            FakeBuilding("high", cls=3, conf=0.95, caption="destroyed"),
            FakeBuilding("low", cls=3, conf=0.20, caption="destroyed"),
            FakeBuilding("contradicted", cls=0, conf=0.99, caption="roof collapsed"),
        ]
    )
    assert [b.footprint_id for b in ordered] == ["contradicted", "low", "high"]


# ------------------------------------------------------- the honesty machinery
def test_spread_check_calls_a_column_of_floors_degenerate(store, monkeypatch):
    install(monkeypatch, Recorder(["3"] * 8 * 12))
    results = ballot.vote_batch([FakeBuilding(f"b{i}", caption="c") for i in range(12)], k=8)

    check = ballot.spread_check(results)

    assert check["total"] == 12 and check["model"] == 12
    assert check["at_floor"] == 12 and check["contested"] == 0
    assert check["floor_share"] == 1.0
    assert check["unanimous"] == 12
    assert check["mean_agreement"] == 1.0
    assert check["degenerate"] is True
    assert "temperature" in check["fallback"], "the remedy is named with the verdict"
    assert str(ballot.ESCALATE_TOP_N) in check["fallback"]
    assert "degenerate" in ballot.distribution_note()


def test_spread_check_reports_a_real_spread_as_healthy(store, monkeypatch):
    """Four buildings genuinely contested, eight unanimous.

    Keyed on each building's own caption, because a sweep interleaves 96 samples
    across the pool: order-keyed replies would assign votes by thread scheduling
    and this assertion would be a coin flip.
    """
    split = ["3", "3", "2", "2", "1", "1", "0", "0"]
    buildings = [FakeBuilding(f"b{i}", caption=f"caption for b{i}") for i in range(12)]
    scripted = {
        f"caption for b{i}": (split if i % 3 == 0 else ["3"] * 8) for i in range(12)
    }
    install(monkeypatch, Recorder(scripted=scripted))

    results = ballot.vote_batch(buildings, k=8)

    check = ballot.spread_check(results)
    assert check["model"] == 12, "every ballot must have taken the model path"
    assert check["contested"] == 4 and check["at_floor"] == 8
    assert check["degenerate"] is False
    assert check["fallback"] == "distribution has spread, no remedy needed"


def test_spread_check_refuses_to_judge_a_thin_sample(store, monkeypatch):
    install(monkeypatch, Recorder(["3"] * 16))
    results = ballot.vote_batch([FakeBuilding(f"b{i}", caption="c") for i in range(2)], k=8)
    check = ballot.spread_check(results)
    assert check["degenerate"] is False
    assert "below the" in check["fallback"]


def test_escalate_revotes_the_least_certain_at_a_higher_temperature(store, monkeypatch):
    """The remedy targets the least certain by OUR inputs, not by the flat ballot."""
    buildings = [
        FakeBuilding(f"b{i}", conf=0.5 + i * 0.04, caption=f"caption for b{i}")
        for i in range(12)
    ]
    rec = install(monkeypatch, Recorder(["3"] * 8 * 12))
    first = ballot.vote_batch(buildings, k=8)
    assert ballot.spread_check(first)["degenerate"] is True

    # The three lowest grader confidences are b0, b1, b2, so the remedy takes them.
    split = ["3", "3", "3", "2", "2", "1", "1", "0"]
    revoted = install(
        monkeypatch,
        Recorder(scripted={f"caption for b{i}": list(split) for i in range(3)}),
    )
    merged = ballot.escalate(buildings, first, top_n=3, k=8)

    assert {c["kw"]["temperature"] for c in revoted.calls} == {ballot.ESCALATE_TEMPERATURE}
    assert len(revoted.calls) == 24, "three buildings re-voted at k=8"
    assert len(merged) == len(first)
    contested = {r.footprint_id for r in merged if not r.at_floor}
    assert contested == {"b0", "b1", "b2"}, "only the least certain rows were re-voted"


def test_spread_check_counts_cross_model_agreement(store, monkeypatch):
    """B8 publishes self-agreement AND agreement with the grader. Two numbers."""
    install(
        monkeypatch,
        Recorder(
            scripted={
                "grader agrees": ["2"] * 8,
                "grader differs": ["3"] * 8,
            }
        ),
    )
    results = ballot.vote_batch(
        [
            FakeBuilding("matches", cls=2, caption="grader agrees"),
            FakeBuilding("differs", cls=1, caption="grader differs"),
        ],
        k=8,
    )
    check = ballot.spread_check(results)
    assert check["model"] == 2
    assert check["agrees_with_grader"] == 1
    by_id = {r.footprint_id: r for r in results}
    assert by_id["matches"].voted_class == 2 and by_id["matches"].grader_class == 2
    assert by_id["differs"].voted_class == 3 and by_id["differs"].grader_class == 1


def test_distribution_note_only_quotes_measured_numbers(store, monkeypatch):
    assert "no ballot has run yet" in ballot.distribution_note()
    install(monkeypatch, Recorder(["3"] * 6 + ["2", "2"]))
    ballot.vote_batch([FakeBuilding("b1", caption="c")], k=8)
    note = ballot.distribution_note()
    assert "k=8 ballot on 1 buildings" in note
    assert "mean doubt 0.25" in note
    assert "mean self-agreement 0.75" in note


# ---------------------------------------------------------------- tag sweep
def test_extract_tags_is_index_aligned_and_sanitized(store, monkeypatch):
    payload = {
        "captions": [
            {"index": 0, "tags": ["collapsed roof", "Standing Water", "collapsed roof"]},
            {"index": 1, "tags": ["two people on the roof", "flooded street"]},
        ]
    }
    install(monkeypatch, Recorder([json.dumps(payload)]))

    tags = ballot.extract_tags(["roof collapsed", "flooding in the street"])

    assert len(tags) == 2
    assert tags[0] == ["collapsed roof", "standing water"], "lowercased and deduped"
    assert "two people on the roof" not in tags[1], "person language never becomes a tag"
    assert "flooded street" in tags[1]


def test_extract_tags_falls_back_per_chunk(store, monkeypatch):
    install(monkeypatch, Recorder([], how=vlm.GRADE_HOW_STUB))
    tags = ballot.extract_tags(["the building is on fire", "nothing notable here"])
    assert len(tags) == 2
    assert "fire" in tags[0], "the deterministic vocabulary answers when the model does not"


# ------------------------------------------------------------ the agency plan
FIRE = contracts.AGENCIES.index("fire")
EMS = contracts.AGENCIES.index("ems")
COLLAPSE = planner.TASK_VOCAB.index("collapse search, possible entrapment")
WELFARE = planner.TASK_VOCAB.index("welfare check and evacuation support")


# The reply is positional, so a test that hardcodes positions would silently test
# the wrong building the moment the scorer reorders. These helpers derive the
# order from the rank the planner was actually given.
ASSIGNMENTS = {
    "fire1": (FIRE, COLLAPSE, 3),
    "ems1": (EMS, WELFARE, 2),
}


def _plan_reply(items, overrides=None) -> str:
    """One [agency, task, units] triple per ranked building, in RANK order."""
    over = overrides or {}
    rows = []
    for it in items:
        fid = it["footprint_id"]
        rows.append(list(over.get(fid, ASSIGNMENTS[fid])))
    return json.dumps({"a": rows})


def _position_of(items, footprint_id: str) -> int:
    return [it["footprint_id"] for it in items].index(footprint_id)


def _seed_plan_corpus() -> list[dict]:
    """Two buildings: a destroyed structure and a nursing home.

    Rank order is whatever the scorer says, and the helpers above follow it, so
    this fixture keeps testing the plan rather than the sort.
    """
    add_building("fire1", cls=3, conf=0.9, label="1200 Gulf Blvd", last_seen_at=1.0)
    add_building(
        "ems1",
        cls=2,
        conf=0.8,
        label="Providence Mount",
        facility={"name": "Providence Mount", "type": "nursing_home", "dist_m": 40},
        last_seen_at=1.0,
    )
    db.run(
        "INSERT INTO availability (agency, units_available, operator, ts) VALUES (?,?,?,?)",
        ("fire", 2, "chief", 1.0),
    )
    items = scorer.rank(limit=12)["items"]
    assert {it["footprint_id"] for it in items} == {"fire1", "ems1"}
    return items


def test_plan_draft_matches_the_section_seven_contract_exactly(store, monkeypatch):
    items = _seed_plan_corpus()
    install(monkeypatch, Recorder([_plan_reply(items)]))

    out = planner.draft_plan(items, {"fire": 2, "ems": 4}, force_invalid_first=False)

    assert out["drafted_by"] == planner.DRAFTED_BY_MODEL
    assert out["recovery"] is None, "a first attempt that was already valid is not a recovery"
    assert [a["agency"] for a in out["agencies"]] == list(contracts.AGENCIES)
    for entry in out["agencies"]:
        assert set(entry) == {"agency", "units_required", "units_available", "steps"}, (
            "route is ABSENT until B4, and a null would read as 'routed, empty' on the map"
        )
        assert isinstance(entry["units_required"], int)
        assert isinstance(entry["units_available"], int)
        for n, step in enumerate(entry["steps"], start=1):
            assert set(step) == {"n", "footprint_id", "label", "centroid", "task", "units"}
            assert step["n"] == n, "numerals are dense and one-based per agency"
            assert isinstance(step["units"], int) and step["units"] >= 1
            assert len(step["centroid"]) == 2

    fire = next(a for a in out["agencies"] if a["agency"] == "fire")
    assert fire["units_required"] == 3 and fire["units_available"] == 2
    assert fire["steps"][0]["footprint_id"] == "fire1"
    assert fire["steps"][0]["label"] == "1200 Gulf Blvd", "the label is OUR data, not the model's"
    assert fire["steps"][0]["centroid"] == CENTROID
    assert fire["steps"][0]["task"] == "collapse search, possible entrapment"

    ems = next(a for a in out["agencies"] if a["agency"] == "ems")
    assert ems["steps"][0]["footprint_id"] == "ems1"
    assert ems["units_required"] == 2 and ems["units_available"] == 4


def test_the_model_cannot_name_a_building_or_write_a_task_string(store, monkeypatch):
    """The model picks indices, so a hallucinated id or task string cannot exist.

    A dispatch card only ever renders an id and a centroid WE supplied and a task
    string from TASK_VOCAB, which also means a hostile caption cannot put text on
    a crew's printed sheet.
    """
    items = _seed_plan_corpus()
    install(monkeypatch, Recorder([_plan_reply(items)]))

    out = planner.draft_plan(items, {"fire": 2}, force_invalid_first=False)

    tasks = [s["task"] for a in out["agencies"] for s in a["steps"]]
    assert tasks, "the plan must have steps"
    assert all(t in planner.TASK_VOCAB for t in tasks)
    ids = [s["footprint_id"] for a in out["agencies"] for s in a["steps"]]
    assert set(ids) <= {"fire1", "ems1"}
    for a in out["agencies"]:
        for s in a["steps"]:
            assert s["centroid"] == CENTROID


def test_an_off_menu_index_is_a_validation_error_not_a_silent_drop(store, monkeypatch):
    items = _seed_plan_corpus()
    rec = install(
        monkeypatch,
        Recorder(
            [
                _plan_reply(items, {"fire1": (len(contracts.AGENCIES), COLLAPSE, 3)}),
                _plan_reply(items),
            ]
        ),
    )

    out = planner.draft_plan(items, {"fire": 2}, force_invalid_first=False)

    assert len(rec.calls) == 2
    assert f"agency index {len(contracts.AGENCIES)} is out of range" in rec.prompts[1]
    fire = next(a for a in out["agencies"] if a["agency"] == "fire")
    assert fire["steps"][0]["footprint_id"] == "fire1"


def test_a_short_array_is_rejected_because_a_partial_plan_looks_complete(store, monkeypatch):
    items = _seed_plan_corpus()
    rec = install(
        monkeypatch,
        Recorder(
            [
                json.dumps({"a": [[FIRE, COLLAPSE, 3]]}),
                _plan_reply(items),
            ]
        ),
    )

    out = planner.draft_plan(items, {"fire": 2}, force_invalid_first=False)

    assert len(rec.calls) == 2
    assert "had 1 entries but there are 2 buildings" in rec.prompts[1]
    assert out["drafted_by"] == planner.DRAFTED_BY_MODEL
    assert sum(len(a["steps"]) for a in out["agencies"]) == 2, "every building is covered"


def test_no_action_is_an_explicit_decision_not_an_omission(store, monkeypatch):
    """"The model judged it clear" and "the model forgot it" must be different."""
    items = _seed_plan_corpus()
    install(
        monkeypatch,
        Recorder([_plan_reply(items, {"ems1": (EMS, planner.TASK_NO_ACTION, 1)})]),
    )

    out = planner.draft_plan(items, {"fire": 2}, force_invalid_first=False)

    assert out["drafted_by"] == planner.DRAFTED_BY_MODEL
    ems = next(a for a in out["agencies"] if a["agency"] == "ems")
    assert ems["steps"] == [] and ems["units_required"] == 0
    fire = next(a for a in out["agencies"] if a["agency"] == "fire")
    assert len(fire["steps"]) == 1


def test_an_all_no_action_plan_is_rejected(store, monkeypatch):
    items = _seed_plan_corpus()
    n = planner.TASK_NO_ACTION
    rec = install(
        monkeypatch,
        Recorder([_plan_reply(items, {"fire1": (FIRE, n, 1), "ems1": (EMS, n, 1)})] * 2),
    )

    out = planner.draft_plan(items, {"fire": 2}, force_invalid_first=False)

    assert len(rec.calls) == 2
    assert "every building was marked no action" in rec.prompts[1]
    assert out["drafted_by"] == planner.DRAFTED_BY_STUB


# ------------------------------------------------------------- self-recovery
def test_invalid_first_response_triggers_exactly_one_reprompt_with_the_error(store, monkeypatch):
    items = _seed_plan_corpus()
    rec = install(
        monkeypatch,
        Recorder(
            [
                _plan_reply(items, {"fire1": (FIRE, COLLAPSE, 99)}),
                _plan_reply(items),
            ]
        ),
    )

    out = planner.draft_plan(items, {"fire": 2}, force_invalid_first=False)

    assert len(rec.calls) == 2, "exactly one re-prompt, never a retry loop"
    assert out["attempts"] == 2
    assert out["recovery"] == planner.RECOVERY_MODEL, "the HUD must read 'model recovered'"
    assert out["drafted_by"] == planner.DRAFTED_BY_MODEL
    retry = rec.prompts[1]
    assert "rejected by the schema validator" in retry
    assert "'units' must be between 1 and 20, got 99" in retry, (
        "the validation error text goes back verbatim"
    )
    assert out["validation_errors"] and "got 99" in out["validation_errors"][0]


def test_two_invalid_responses_fall_back_with_an_honest_drafted_by(store, monkeypatch):
    items = _seed_plan_corpus()
    rec = install(monkeypatch, Recorder(["not json at all", '{"a": []}']))

    out = planner.draft_plan(items, {"fire": 2}, force_invalid_first=False)

    assert len(rec.calls) == 2, "one attempt plus one re-prompt, then stop"
    assert out["drafted_by"] == planner.DRAFTED_BY_STUB
    assert out["recovery"] == planner.RECOVERY_STUB, "the HUD must read 'stub engaged'"
    assert out["agencies"], "the panel is never empty: the rule set answered"
    assert [a["agency"] for a in out["agencies"]] == list(contracts.AGENCIES)
    for entry in out["agencies"]:
        assert set(entry) == {"agency", "units_required", "units_available", "steps"}
    fire = next(a for a in out["agencies"] if a["agency"] == "fire")
    assert fire["units_available"] == 2, "operator-entered availability survives the fallback"


def test_a_dead_endpoint_does_not_burn_the_reprompt(store, monkeypatch):
    """The re-prompt exists to fix a schema error. An unreachable port is not one."""
    items = _seed_plan_corpus()
    rec = install(monkeypatch, Recorder([], how=vlm.GRADE_HOW_STUB))

    out = planner.draft_plan(items, {"fire": 2}, force_invalid_first=False)

    assert len(rec.calls) == 1
    assert out["drafted_by"] == planner.DRAFTED_BY_STUB
    assert out["recovery"] == planner.RECOVERY_STUB


def test_force_invalid_first_fires_the_recovery_path_deterministically(store, monkeypatch):
    items = _seed_plan_corpus()
    rec = install(monkeypatch, Recorder([_plan_reply(items)] * 4))

    for _ in range(3):
        out = planner.draft_plan(items, {"fire": 2}, force_invalid_first=True)
        assert out["recovery"] == planner.RECOVERY_MODEL, "the beat fires every time"
        assert out["attempts"] == 2
        assert out["drafted_by"] == planner.DRAFTED_BY_MODEL
        assert out["forced_invalid_first"] is True

    # The forced attempt costs no round trip: only the retries reached the model.
    assert len(rec.calls) == 3
    for prompt in rec.prompts:
        assert "rejected by the schema validator" in prompt
        assert "out of range" in prompt, (
            "the forced attempt must be a CONTRACT violation the validator can name, "
            "not unparseable garbage"
        )


def test_force_invalid_first_reads_the_env_flag(store, monkeypatch):
    items = _seed_plan_corpus()
    monkeypatch.setenv("FIRSTLIGHT_DEMO_FORCE_INVALID", "1")
    assert planner.force_invalid_first() is True
    install(monkeypatch, Recorder([_plan_reply(items)]))
    out = planner.draft_plan(items, {"fire": 2})
    assert out["recovery"] == planner.RECOVERY_MODEL
    monkeypatch.setenv("FIRSTLIGHT_DEMO_FORCE_INVALID", "off")
    assert planner.force_invalid_first() is False


def test_recovery_is_recorded_in_the_append_only_log(store, monkeypatch):
    items = _seed_plan_corpus()
    install(monkeypatch, Recorder([_plan_reply(items)]))
    planner.draft_plan(items, {"fire": 2}, force_invalid_first=True, operator="chief")
    row = db.q1("SELECT actor, action, payload FROM decision_log ORDER BY id DESC LIMIT 1")
    payload = json.loads(row["payload"])
    assert row["action"] == "plan-drafted"
    assert payload["recovery"] == planner.RECOVERY_MODEL
    assert payload["forced_invalid_first"] is True
    assert payload["drafted_by"] == planner.DRAFTED_BY_MODEL


# ------------------------------------------------------------- next flight
def _flight_reply(**kw) -> str:
    spec = {
        "sector_center": [-82.74, 27.78],
        "half_width_deg": 0.008,
        "half_height_deg": 0.006,
        "altitude_m_agl": 90,
        "line_spacing_m": 60,
        "transects": 7,
        "est_flight_min": 22,
        "reason": "sector C has gone longest without a look and the arterial is cut",
    }
    spec.update(kw)
    return json.dumps(spec)


def test_next_flight_matches_the_flight_plan_contract(store, monkeypatch):
    items = _seed_plan_corpus()
    install(monkeypatch, Recorder([_flight_reply()]))

    fc = planner.next_flight(items, ["Gulf Blvd"], force_invalid_first=False)

    assert fc["type"] == "FeatureCollection" and len(fc["features"]) == 2
    roles = [f["properties"]["role"] for f in fc["features"]]
    assert roles == ["survey-area", "survey-path"]

    area = fc["features"][0]
    assert area["geometry"]["type"] == "Polygon"
    ring = area["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1] and len(ring) == 5

    path = fc["features"][1]
    assert path["geometry"]["type"] == "LineString"
    props = path["properties"]
    for key in ("altitude_m_agl", "line_spacing_m", "transects", "est_flight_min"):
        assert key in props
    assert props["transects"] == 7
    assert len(path["geometry"]["coordinates"]) == 14, "two points per transect"
    assert props["drafted_by"] == planner.DRAFTED_BY_MODEL


def test_the_survey_path_actually_serpentines(store, monkeypatch):
    """Geometric truth: a pattern that jumps back to the same side is a drawing."""
    install(monkeypatch, Recorder([_flight_reply(transects=5)]))
    fc = planner.next_flight([], [], force_invalid_first=False)
    coords = fc["features"][1]["geometry"]["coordinates"]
    legs = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
    directions = [1 if b[0] > a[0] else -1 for a, b in legs]
    assert directions == [1, -1, 1, -1, 1], "every transect reverses"
    ys = [a[1] for a, _ in legs]
    assert ys == sorted(ys), "the pattern advances in one direction"


def test_a_flight_box_outside_the_area_of_operations_is_clamped(store, monkeypatch):
    install(monkeypatch, Recorder([_flight_reply(sector_center=[-82.60, 27.90])]))
    fc = planner.next_flight([], [], force_invalid_first=False)
    w, s, e, n = AOI
    for lng, lat in fc["features"][0]["geometry"]["coordinates"][0]:
        assert w - 1e-9 <= lng <= e + 1e-9
        assert s - 1e-9 <= lat <= n + 1e-9


def test_a_latitude_in_the_longitude_slot_is_a_validation_error(store, monkeypatch):
    rec = install(
        monkeypatch,
        Recorder([_flight_reply(sector_center=[27.78, -182.0]), _flight_reply()]),
    )
    fc = planner.next_flight([], [], force_invalid_first=False)
    assert len(rec.calls) == 2
    assert "out of range" in rec.prompts[1] and "[lng, lat] order" in rec.prompts[1]
    assert fc["features"][0]["properties"]["recovery"] == planner.RECOVERY_MODEL


def test_a_whitespace_padded_body_still_parses(store, monkeypatch):
    """Measured on the box: the json_schema grammar permits unlimited trailing
    whitespace, so one flight call in three ran the decoder to max_tokens emitting
    only newlines after a complete object, 42528 ms for a payload valid at 112
    tokens. max_tokens now bounds it, which means a length-truncated tail of
    whitespace reaches the parser, and it must not become a stub."""
    install(monkeypatch, Recorder([_flight_reply() + "\n" * 400]))
    fc = planner.next_flight([], [], force_invalid_first=False)
    assert fc["features"][0]["properties"]["drafted_by"] == planner.DRAFTED_BY_MODEL
    assert fc["features"][0]["properties"]["recovery"] is None


def test_a_body_truncated_mid_string_is_a_named_validation_error(store, monkeypatch):
    """Scanning for any JSON value would return the inner sector_center array here
    and the validator would report a baffling type error. It must name truncation."""
    cut = '{"sector_center": [-82.74, 27.78], "reason": "sector E is stal'
    rec = install(monkeypatch, Recorder([cut, _flight_reply()]))

    fc = planner.next_flight([], [], force_invalid_first=False)

    assert len(rec.calls) == 2
    assert "no complete JSON object" in rec.prompts[1]
    assert "cut off" in rec.prompts[1]
    assert fc["features"][0]["properties"]["recovery"] == planner.RECOVERY_MODEL


def test_token_caps_stay_sized_to_measured_output(store):
    """These caps ARE the latency budget at 24 tok/s, and the only bound on the
    whitespace run. Measured completion tokens: plan 78, flight 94 to 129."""
    assert planner.PLAN_MAX_TOKENS <= 512
    assert planner.FLIGHT_MAX_TOKENS <= 512
    assert planner.RATIONALE_MAX_TOKENS <= 256


def test_flight_falls_back_to_the_stalest_ranked_sector(store, monkeypatch):
    add_building("stale", cls=2, label="Stalest Block", centroid=[-82.75, 27.79], last_seen_at=1.0)
    items = scorer.rank(limit=10)["items"]
    install(monkeypatch, Recorder([], how=vlm.GRADE_HOW_STUB))

    fc = planner.next_flight(items, [], force_invalid_first=False)

    props = fc["features"][0]["properties"]
    assert props["drafted_by"] == planner.DRAFTED_BY_STUB
    assert props["recovery"] == planner.RECOVERY_STUB
    assert "Stalest Block" in props["reason"] and "hours since the last look" in props["reason"]
    assert fc["features"][1]["properties"]["est_flight_min"] > 0


def test_force_invalid_first_fires_the_flight_recovery_too(store, monkeypatch):
    rec = install(monkeypatch, Recorder([_flight_reply()] * 3))
    for _ in range(2):
        fc = planner.next_flight([], [], force_invalid_first=True)
        assert fc["features"][0]["properties"]["recovery"] == planner.RECOVERY_MODEL
        assert fc["features"][1]["properties"]["forced_invalid_first"] is True
    assert len(rec.calls) == 2, "the forced attempt costs no round trip"


def test_the_thinking_trace_is_the_models_own_prose_or_empty(store, monkeypatch):
    install(monkeypatch, Recorder(["Sector C is stalest and cut off. " + _flight_reply()]))
    fc = planner.next_flight([], [], force_invalid_first=False)
    assert fc["features"][1]["properties"]["thinking"] == "Sector C is stalest and cut off."

    install(monkeypatch, Recorder([_flight_reply()]))
    fc = planner.next_flight([], [], force_invalid_first=False)
    assert fc["features"][1]["properties"]["thinking"] == "", "never invent a trace"


# ------------------------------------------------------------- hero rationale
def test_hero_rationale_prompt_carries_only_scorer_inputs(store, monkeypatch):
    _seed_plan_corpus()
    item = scorer.rank(limit=1)["items"][0]
    rec = install(monkeypatch, Recorder(["It is first because nobody has looked in hours."]))

    text, by = planner.hero_rationale(item)

    assert by == planner.RATIONALE_BY_NANO
    assert text == "It is first because nobody has looked in hours."
    prompt = rec.prompts[0]
    for key in ("severity_weight", "staleness_h", "vulnerable_density", "doubt"):
        assert str(item["inputs"][key]) in prompt, f"{key} must be citable"
    assert str(item["priority"]) in prompt
    assert "/no_think" in prompt


def test_the_fallback_rationale_cites_the_same_inputs(store, monkeypatch):
    _seed_plan_corpus()
    item = scorer.rank(limit=1)["items"][0]
    install(monkeypatch, Recorder([], how=vlm.GRADE_HOW_STUB))

    text, by = planner.hero_rationale(item)

    assert by == planner.DRAFTED_BY_STUB
    for key in ("staleness_h", "vulnerable_density", "doubt"):
        assert str(item["inputs"][key]) in text, "B8 checks cited inputs against actual inputs"
    assert str(item["priority"]) in text


def test_batch_rationales_label_themselves_honestly(store, monkeypatch):
    _seed_plan_corpus()
    items = scorer.rank(limit=12)["items"]
    rec = install(monkeypatch, Recorder([]))

    out = planner.batch_rationales(items)

    assert len(out) == len(items)
    assert all(by == planner.DRAFTED_BY_STUB for _, by in out)
    assert not rec.calls, "no generation happened, so no model byline is claimed"


# --------------------------------------------------------------- the replan beat
def test_replan_issues_the_two_calls_concurrently(store, monkeypatch):
    """Both calls are decode bound and share nothing, so the beat must cost the
    slower of the two rather than the sum. Measured on the box: 3.4 s plan plus
    5.0 s flight is 8.5 s in series, which cannot meet the 3 s target."""
    items = _seed_plan_corpus()
    overlap = {"in_flight": 0, "max": 0}
    lock = threading.Lock()
    gate = threading.Barrier(2, timeout=5)

    def slow_chat(endpoint, messages, **kw):
        with lock:
            overlap["in_flight"] += 1
            overlap["max"] = max(overlap["max"], overlap["in_flight"])
        try:
            gate.wait()  # deadlocks unless both calls really are in flight
        finally:
            with lock:
                overlap["in_flight"] -= 1
        body = messages[-1]["content"]
        if "survey box" in body or "Blocked roads" in body:
            return _flight_reply(), vlm.GRADE_HOW_MODEL
        return _plan_reply(items), vlm.GRADE_HOW_MODEL

    monkeypatch.setattr(vlm, "chat", slow_chat)

    out = planner.replan(items, {"fire": 2}, ["Gulf Blvd"], force_invalid_first=False)

    assert overlap["max"] == 2, "the plan and the flight must be in flight together"
    assert out["plan"]["drafted_by"] == planner.DRAFTED_BY_MODEL
    assert out["flight"]["features"][0]["properties"]["role"] == "survey-area"
    assert out["recovery"] is None
    assert out["took_ms"] >= 0


def test_replan_recovery_reports_the_worse_of_the_two(store, monkeypatch):
    """One HUD indicator, so a beat where half the agent fell back reads as stub."""
    items = _seed_plan_corpus()

    def half_dead(endpoint, messages, **kw):
        body = messages[-1]["content"]
        if "survey box" in body or "Blocked roads" in body:
            return "", vlm.GRADE_HOW_STUB
        return _plan_reply(items), vlm.GRADE_HOW_MODEL

    monkeypatch.setattr(vlm, "chat", half_dead)

    out = planner.replan(items, {"fire": 2}, [], force_invalid_first=False)

    assert out["plan"]["drafted_by"] == planner.DRAFTED_BY_MODEL
    assert out["flight"]["features"][0]["properties"]["drafted_by"] == planner.DRAFTED_BY_STUB
    assert out["recovery"] == planner.RECOVERY_STUB
    assert planner.last_recovery() == planner.RECOVERY_STUB


def test_replan_forces_both_recovery_beats_together(store, monkeypatch):
    items = _seed_plan_corpus()

    def ok(endpoint, messages, **kw):
        body = messages[-1]["content"]
        assert "rejected by the schema validator" in body, "both calls must be retries"
        if "survey box" in body or "Blocked roads" in body:
            return _flight_reply(), vlm.GRADE_HOW_MODEL
        return _plan_reply(items), vlm.GRADE_HOW_MODEL

    monkeypatch.setattr(vlm, "chat", ok)

    out = planner.replan(items, {"fire": 2}, [], force_invalid_first=True)

    assert out["recovery"] == planner.RECOVERY_MODEL
    assert out["plan"]["recovery"] == planner.RECOVERY_MODEL
    assert out["flight"]["features"][0]["properties"]["recovery"] == planner.RECOVERY_MODEL


# ------------------------------------------------------------------- measured
def test_replan_percentiles_and_model_versions_report_the_path_that_ran(store, monkeypatch):
    assert planner.replan_p95() == 0 and planner.last_replan_ms() == 0
    assert "planner idle" in planner.model_version()
    assert "ballot idle" in ballot.model_version()

    items = _seed_plan_corpus()
    install(monkeypatch, Recorder([], how=vlm.GRADE_HOW_STUB))
    planner.draft_plan(items, {"fire": 2}, force_invalid_first=False)

    assert planner.last_replan_ms() >= 0
    assert planner.replan_p95() >= planner.replan_p50()
    assert "stub rules engaged" in planner.model_version()
    assert planner.last_recovery() == planner.RECOVERY_STUB

    install(monkeypatch, Recorder(["3"] * 6 + ["2", "2"]))
    ballot.vote_batch([FakeBuilding("b1", caption="c")], k=8)
    version = ballot.model_version()
    assert "k=8 ballot" in version and "measured" in version


def _owner(plan: dict, fid: str):
    for entry in plan["agencies"]:
        for step in entry["steps"]:
            if step["footprint_id"] == fid or fid in (step.get("footprint_ids") or []):
                return entry["agency"], step["n"]
    return None, None


def test_a_reassign_survives_the_next_plan_poll(store):
    """An operator edit must outlive the re-draft that happens two seconds later.

    build_plan recomputes from the live ranking on every poll, so an edit recorded
    only in the decision log was gone by the next refresh: the console showed the
    reassign, then snapped the stop back to its drafted agency in front of the
    operator. This is the regression guard for that, and it polls twice because
    once would pass even with the edit held only in memory.
    """
    add_building("fp_1", cls=3, label="801 W 13TH ST", centroid=[-85.67, 30.17])
    drafted_agency, _ = _owner(scorer.build_plan(), "fp_1")
    assert drafted_agency is not None, "the fixture building must reach the plan"

    target = "ems" if drafted_agency != "ems" else "police"
    scorer.set_plan_override("fp_1", operator="R. Alvarez", agency=target)

    for poll in range(3):
        agency, _n = _owner(scorer.build_plan(), "fp_1")
        assert agency == target, f"poll {poll + 1} reverted the reassign to {agency}"


def test_a_deleted_stop_stays_deleted_and_reset_brings_it_back(store):
    """Delete must persist, and the operator must be able to get the draft back."""
    add_building("fp_1", cls=3, label="801 W 13TH ST", centroid=[-85.67, 30.17])
    assert _owner(scorer.build_plan(), "fp_1")[0] is not None

    scorer.set_plan_override("fp_1", operator="R. Alvarez", deleted=True)
    assert _owner(scorer.build_plan(), "fp_1")[0] is None
    assert _owner(scorer.build_plan(), "fp_1")[0] is None  # and again

    # An operator who over-edits needs a way back to the drafted plan, otherwise a
    # mistaken delete is unrecoverable mid-incident.
    assert scorer.clear_plan_overrides("R. Alvarez") == 1
    assert _owner(scorer.build_plan(), "fp_1")[0] is not None


def test_overrides_compose_rather_than_overwriting_each_other(store):
    """Reassign then reorder must not drop the reassign.

    The edits arrive as separate requests keyed on the same footprint, so a plain
    INSERT OR REPLACE that took only the newest field would silently undo the
    earlier decision.
    """
    add_building("fp_1", cls=3, label="801 W 13TH ST", centroid=[-85.67, 30.17])
    add_building("fp_2", cls=3, label="705 W 15TH ST", centroid=[-85.671, 30.171])

    scorer.set_plan_override("fp_1", operator="R. Alvarez", agency="police")
    scorer.set_plan_override("fp_1", operator="R. Alvarez", order_key=0)
    scorer.set_plan_override("fp_1", operator="R. Alvarez", units=4)

    agency, _n = _owner(scorer.build_plan(), "fp_1")
    assert agency == "police", "the later order/units edits dropped the reassign"
    step = next(
        s
        for e in scorer.build_plan()["agencies"]
        for s in e["steps"]
        if s["footprint_id"] == "fp_1"
    )
    assert step["units"] == 4


def test_an_unknown_agency_is_refused_rather_than_stored(store):
    """A typo must not create a fifth agency that no crew belongs to."""
    add_building("fp_1", cls=3, centroid=[-85.67, 30.17])
    with pytest.raises(ValueError):
        scorer.set_plan_override("fp_1", operator="R. Alvarez", agency="coastguard")
    with pytest.raises(ValueError):
        scorer.set_plan_override("fp_1", operator="   ", agency="ems")


def test_tied_priorities_keep_a_stable_order_across_reranks(store):
    """Equal inputs tie, and the tie must not reshuffle between polls.

    Three buildings on one street with the same class, staleness and doubt produce
    the same priority - that is the formula being deterministic, not a bug. But the
    sort key was (confirmed-severe, -priority) with no final tiebreak, so the order
    among tied rows came from whatever sequence SQLite returned. An operator reading
    "second on the list" and looking again after a re-rank could see a different
    building there.
    """
    for fid in ("fp_c", "fp_a", "fp_b"):
        add_building(fid, cls=2, conf=0.75, centroid=[-85.67, 30.17], last_seen_at=1.0)

    first = [it["footprint_id"] for it in scorer.rank(limit=10)["items"]]
    assert len(first) == 3

    # Tied on every input, so the tiebreak is the only thing ordering them.
    priorities = {round(it["priority"], 6) for it in scorer.rank(limit=10)["items"]}
    assert len(priorities) == 1, f"expected a genuine tie, got {priorities}"

    for _ in range(3):
        again = [it["footprint_id"] for it in scorer.rank(limit=10)["items"]]
        assert again == first, f"tied rows reshuffled: {first} then {again}"
