#!/usr/bin/env python3
import argparse
import json
import math
import re
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


LINE_TS = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) (?P<msg>.*)$")
LEAGUE_START = re.compile(r"=== .*?(\d+) 智能体联赛启动 ===")
GAME_START = re.compile(
    r"\[对局 #(?P<id>\d+) 开始\] Agent_(?P<a>\d+)\(ELO:(?P<aelo>-?\d+(?:\.\d+)?) DNA:S(?P<asim>\d+)/C(?P<acpuct>\d+(?:\.\d+)?)/T(?P<atemp>\d+(?:\.\d+)?)\) "
    r"VS Agent_(?P<b>\d+)\(ELO:(?P<belo>-?\d+(?:\.\d+)?) DNA:S(?P<bsim>\d+)/C(?P<bcpuct>\d+(?:\.\d+)?)/T(?P<btemp>\d+(?:\.\d+)?)\)"
)
GAME_END = re.compile(
    r"\[对局 #(?P<id>\d+) 结束\] Agent_(?P<a>\d+)\(ELO:(?P<aelo>-?\d+(?:\.\d+)?)\) "
    r"VS Agent_(?P<b>\d+)\(ELO:(?P<belo>-?\d+(?:\.\d+)?)\) \| (?P<result>[^|]+) \| (?P<moves>\d+)步"
)
GAME_TIMEOUT = re.compile(r"\[对局 #(?P<id>\d+) 超时\] Agent_(?P<a>\d+) VS Agent_(?P<b>\d+) .* 已走 (?P<moves>\d+) 步")
TRAIN_START = re.compile(r"\[周期训练\] 开始：.*?大师样本 (?P<master>\d+)，联赛样本 (?P<league>\d+)，(?:候选智能体|训练智能体) (?P<agents>\d+)")
TRAIN_DONE = re.compile(r"\[周期训练\] 完成：训练 (?P<agents>\d+) 个智能体，使用 (?P<samples>\d+) 条样本，平均损失 (?P<loss>-?\d+(?:\.\d+)?)")
TRAIN_CLEAN = re.compile(r"\[周期训练\] 联赛样本清理：保留最近 (?P<games>\d+) 局（(?P<samples>\d+) 条），删除 (?P<deleted>\d+) 局旧对局。")
TOP_ID = re.compile(r"ID:(?P<id>\d+) ELO:(?P<elo>-?\d+(?:\.\d+)?) 胜率:(?P<wr>\d+(?:\.\d+)?)%")
COUNTER = re.compile(r"\[(?P<name>[A-Za-z][A-Za-z0-9]+)(?:/[0-9a-f]+)?\] samples=(?P<samples>\d+) avg=(?P<avg>-?\d+(?:\.\d+)?) max=(?P<max>-?\d+(?:\.\d+)?)")
EVAL_START = re.compile(
    r"Evaluation started: candidate=(?P<candidate>[^,]+), baseline=(?P<baseline>[^,]+), games=(?P<games>\d+), sims=(?P<sims>\d+), c_puct=(?P<cpuct>\d+(?:\.\d+)?), root_noise=(?P<noise>true|false)"
)
EVAL_RESULT = re.compile(
    r"\[Eval (?P<idx>\d+)/(?P<total>\d+)\] result=(?P<result>[^,]+), plies=(?P<plies>\d+), reason=(?P<reason>.*?), game_time=(?P<minutes>\d+(?:\.\d+)?)m, score=(?P<score>\d+(?:\.\d+)?)/(?P<played>\d+) \((?P<pct>\d+(?:\.\d+)?)%\)"
)


def parse_ts(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S.%f")


@dataclass
class AgentStats:
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    final_elo: float | None = None
    max_elo: float | None = None
    min_elo: float | None = None
    sim_values: list[int] = field(default_factory=list)
    cpuct_values: list[float] = field(default_factory=list)
    temp_values: list[float] = field(default_factory=list)

    def note_elo(self, elo: float) -> None:
        self.final_elo = elo
        self.max_elo = elo if self.max_elo is None else max(self.max_elo, elo)
        self.min_elo = elo if self.min_elo is None else min(self.min_elo, elo)


def pct(num: float, den: float) -> float:
    return 0.0 if den == 0 else num * 100.0 / den


def mean(values):
    return None if not values else sum(values) / len(values)


def median(values):
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def analyze(log_path: Path, tail_games: int):
    totals = {
        "lines": 0,
        "first_timestamp": None,
        "last_timestamp": None,
        "league_starts": 0,
        "population_sizes": Counter(),
        "games_started": 0,
        "games_completed": 0,
        "games_timed_out": 0,
        "training_runs": 0,
        "training_failures": 0,
        "game_failures": 0,
    }
    result_counts = Counter()
    move_counts = []
    game_minutes = []
    agent_stats = defaultdict(AgentStats)
    active_games = {}
    tail_completed = deque(maxlen=tail_games)
    training_runs = []
    pending_train = None
    top_snapshots = []
    current_top = []
    errors = Counter()
    counter_latest = {}
    eval_runs = []
    current_eval = None

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            totals["lines"] += 1
            m = LINE_TS.match(raw.rstrip("\n"))
            if not m:
                continue

            ts = parse_ts(m.group("ts"))
            msg = m.group("msg")
            totals["first_timestamp"] = totals["first_timestamp"] or ts
            totals["last_timestamp"] = ts

            if league := LEAGUE_START.search(msg):
                totals["league_starts"] += 1
                totals["population_sizes"][league.group(1)] += 1
                continue

            if gs := GAME_START.search(msg):
                gid = int(gs.group("id"))
                a = int(gs.group("a"))
                b = int(gs.group("b"))
                totals["games_started"] += 1
                active_games[gid] = (ts, a, b)
                agent_stats[a].sim_values.append(int(gs.group("asim")))
                agent_stats[a].cpuct_values.append(float(gs.group("acpuct")))
                agent_stats[a].temp_values.append(float(gs.group("atemp")))
                agent_stats[b].sim_values.append(int(gs.group("bsim")))
                agent_stats[b].cpuct_values.append(float(gs.group("bcpuct")))
                agent_stats[b].temp_values.append(float(gs.group("btemp")))
                agent_stats[a].note_elo(float(gs.group("aelo")))
                agent_stats[b].note_elo(float(gs.group("belo")))
                continue

            if ge := GAME_END.search(msg):
                gid = int(ge.group("id"))
                a = int(ge.group("a"))
                b = int(ge.group("b"))
                result = ge.group("result").strip()
                moves = int(ge.group("moves"))
                totals["games_completed"] += 1
                result_counts[result] += 1
                move_counts.append(moves)
                agent_stats[a].games += 1
                agent_stats[b].games += 1
                if result == "平局":
                    agent_stats[a].draws += 1
                    agent_stats[b].draws += 1
                elif result == "红胜":
                    agent_stats[a].wins += 1
                    agent_stats[b].losses += 1
                elif result == "黑胜":
                    agent_stats[a].losses += 1
                    agent_stats[b].wins += 1
                agent_stats[a].note_elo(float(ge.group("aelo")))
                agent_stats[b].note_elo(float(ge.group("belo")))
                started = active_games.pop(gid, (None, a, b))[0]
                minutes = None
                if started:
                    minutes = (ts - started).total_seconds() / 60.0
                    game_minutes.append(minutes)
                tail_completed.append({
                    "timestamp": ts.isoformat(sep=" "),
                    "game": gid,
                    "agent_a": a,
                    "agent_b": b,
                    "result": result,
                    "moves": moves,
                    "minutes": minutes,
                })
                continue

            if gt := GAME_TIMEOUT.search(msg):
                totals["games_timed_out"] += 1
                active_games.pop(int(gt.group("id")), None)
                continue

            if tr := TRAIN_START.search(msg):
                pending_train = {
                    "timestamp": ts.isoformat(sep=" "),
                    "master_samples": int(tr.group("master")),
                    "league_samples": int(tr.group("league")),
                    "agents": int(tr.group("agents")),
                }
                continue

            if td := TRAIN_DONE.search(msg):
                totals["training_runs"] += 1
                run = pending_train or {"timestamp": ts.isoformat(sep=" ")}
                run.update({
                    "completed_at": ts.isoformat(sep=" "),
                    "trained_agents": int(td.group("agents")),
                    "used_samples": int(td.group("samples")),
                    "loss": float(td.group("loss")),
                })
                training_runs.append(run)
                pending_train = None
                continue

            if tc := TRAIN_CLEAN.search(msg):
                if pending_train is not None:
                    pending_train.update({
                        "retained_league_games": int(tc.group("games")),
                        "retained_league_samples": int(tc.group("samples")),
                        "deleted_league_games": int(tc.group("deleted")),
                    })
                continue

            if top := TOP_ID.search(msg):
                current_top.append({
                    "timestamp": ts.isoformat(sep=" "),
                    "id": int(top.group("id")),
                    "elo": float(top.group("elo")),
                    "win_rate": float(top.group("wr")),
                })
                if len(current_top) == 5:
                    top_snapshots.append(current_top)
                    current_top = []
                continue

            if co := COUNTER.search(msg):
                counter_latest[co.group("name")] = {
                    "timestamp": ts.isoformat(sep=" "),
                    "samples": int(co.group("samples")),
                    "avg": float(co.group("avg")),
                    "max": float(co.group("max")),
                }
                continue

            if evs := EVAL_START.search(msg):
                current_eval = {
                    "timestamp": ts.isoformat(sep=" "),
                    "candidate": evs.group("candidate"),
                    "baseline": evs.group("baseline"),
                    "target_games": int(evs.group("games")),
                    "sims": int(evs.group("sims")),
                    "c_puct": float(evs.group("cpuct")),
                    "root_noise": evs.group("noise") == "true",
                    "results": Counter(),
                    "plies": [],
                    "game_minutes": [],
                    "latest_score": None,
                }
                eval_runs.append(current_eval)
                continue

            if evr := EVAL_RESULT.search(msg):
                if current_eval is None:
                    current_eval = {
                        "timestamp": ts.isoformat(sep=" "),
                        "candidate": None,
                        "baseline": None,
                        "target_games": int(evr.group("total")),
                        "sims": None,
                        "c_puct": None,
                        "root_noise": None,
                        "results": Counter(),
                        "plies": [],
                        "game_minutes": [],
                        "latest_score": None,
                    }
                    eval_runs.append(current_eval)
                result = evr.group("result")
                current_eval["results"][result] += 1
                current_eval["plies"].append(int(evr.group("plies")))
                current_eval["game_minutes"].append(float(evr.group("minutes")))
                current_eval["latest_score"] = {
                    "candidate_score": float(evr.group("score")),
                    "played": int(evr.group("played")),
                    "pct": float(evr.group("pct")),
                    "last_reason": evr.group("reason"),
                }
                continue

            if "[对局异常]" in msg:
                totals["game_failures"] += 1
                errors[msg[:220]] += 1
            elif "[周期训练异常]" in msg:
                totals["training_failures"] += 1
                errors[msg[:220]] += 1
            elif "Exception" in msg or "异常" in msg or "Fatal" in msg or "错误" in msg:
                errors[msg[:220]] += 1

    ranked_agents = []
    for agent_id, stats in agent_stats.items():
        if stats.games == 0 and stats.final_elo is None:
            continue
        ranked_agents.append({
            "id": agent_id,
            "games": stats.games,
            "wins": stats.wins,
            "losses": stats.losses,
            "draws": stats.draws,
            "win_rate": pct(stats.wins, stats.games),
            "non_loss_rate": pct(stats.wins + stats.draws, stats.games),
            "final_elo": stats.final_elo,
            "max_elo": stats.max_elo,
            "min_elo": stats.min_elo,
            "avg_sims": mean(stats.sim_values),
            "avg_cpuct": mean(stats.cpuct_values),
            "avg_temp": mean(stats.temp_values),
        })
    ranked_agents.sort(key=lambda x: (x["final_elo"] if x["final_elo"] is not None else -9999, x["games"]), reverse=True)

    training_losses = [r["loss"] for r in training_runs if "loss" in r]
    eval_summary = []
    for ev in eval_runs:
        played = sum(ev["results"].values())
        latest = ev["latest_score"] or {}
        eval_summary.append({
            "timestamp": ev["timestamp"],
            "candidate": ev["candidate"],
            "baseline": ev["baseline"],
            "target_games": ev["target_games"],
            "played": played,
            "sims": ev["sims"],
            "c_puct": ev["c_puct"],
            "root_noise": ev["root_noise"],
            "results": dict(ev["results"]),
            "candidate_score": latest.get("candidate_score"),
            "candidate_pct": latest.get("pct"),
            "avg_plies": mean(ev["plies"]),
            "avg_game_minutes": mean(ev["game_minutes"]),
            "last_reason": latest.get("last_reason"),
        })

    return {
        "log_path": str(log_path),
        "totals": {
            **{k: (v.isoformat(sep=" ") if isinstance(v, datetime) else v) for k, v in totals.items() if k != "population_sizes"},
            "population_sizes": dict(totals["population_sizes"]),
        },
        "game_results": dict(result_counts),
        "game_quality": {
            "avg_moves": mean(move_counts),
            "median_moves": median(move_counts),
            "max_moves": max(move_counts) if move_counts else None,
            "avg_minutes": mean(game_minutes),
            "median_minutes": median(game_minutes),
            "max_minutes": max(game_minutes) if game_minutes else None,
            "draw_rate": pct(result_counts.get("平局", 0), sum(result_counts.values())),
        },
        "training": {
            "runs": len(training_runs),
            "first_loss": training_losses[0] if training_losses else None,
            "last_loss": training_losses[-1] if training_losses else None,
            "best_loss": min(training_losses) if training_losses else None,
            "avg_loss_last_10": mean(training_losses[-10:]),
            "recent_runs": training_runs[-10:],
        },
        "top_agents": ranked_agents[:15],
        "latest_top_snapshot": top_snapshots[-1] if top_snapshots else [],
        "runtime_counters": counter_latest,
        "eval_runs": eval_summary,
        "recent_games": list(tail_completed),
        "errors": errors.most_common(20),
    }


def write_markdown(report, out_path: Path):
    def fmt(value, spec):
        return "n/a" if value is None else format(value, spec)

    totals = report["totals"]
    game_quality = report["game_quality"]
    training = report["training"]
    lines = []
    lines.append("# ChineseChessAI Runtime Training Report")
    lines.append("")
    lines.append(f"- Log: `{report['log_path']}`")
    lines.append(f"- Range: {totals['first_timestamp']} -> {totals['last_timestamp']}")
    lines.append(f"- Lines parsed: {totals['lines']:,}")
    lines.append(f"- League starts: {totals['league_starts']} population={totals['population_sizes']}")
    lines.append("")
    lines.append("## League Games")
    lines.append("")
    lines.append(f"- Started: {totals['games_started']:,}")
    lines.append(f"- Completed: {totals['games_completed']:,}")
    lines.append(f"- Timed out: {totals['games_timed_out']:,}")
    lines.append(f"- Results: {report['game_results']}")
    lines.append(f"- Draw rate: {game_quality['draw_rate']:.1f}%")
    lines.append(f"- Avg moves: {game_quality['avg_moves']:.1f}" if game_quality["avg_moves"] is not None else "- Avg moves: n/a")
    lines.append(f"- Median moves: {game_quality['median_moves']}")
    lines.append(f"- Max moves: {game_quality['max_moves']}")
    lines.append(f"- Avg game minutes: {game_quality['avg_minutes']:.2f}" if game_quality["avg_minutes"] is not None else "- Avg game minutes: n/a")
    lines.append("")
    lines.append("## Training")
    lines.append("")
    lines.append(f"- Completed training runs: {training['runs']:,}")
    lines.append(f"- Training failures: {totals['training_failures']:,}")
    lines.append(f"- First loss: {training['first_loss']}")
    lines.append(f"- Last loss: {training['last_loss']}")
    lines.append(f"- Best loss: {training['best_loss']}")
    lines.append(f"- Avg loss last 10: {training['avg_loss_last_10']}")
    lines.append("")
    lines.append("Recent training runs:")
    lines.append("")
    for run in training["recent_runs"]:
        lines.append(f"- {run.get('completed_at', run.get('timestamp'))}: loss={run.get('loss')} trained={run.get('trained_agents')} used_samples={run.get('used_samples')} master={run.get('master_samples')} league={run.get('league_samples')}")
    lines.append("")
    lines.append("## Top Agents By Final Elo")
    lines.append("")
    lines.append("| Agent | Elo | Games | W-D-L | Win% | Non-loss% | Avg sims | Avg cpuct | Avg temp |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for a in report["top_agents"]:
        lines.append(
            f"| {a['id']} | {a['final_elo']:.0f} | {a['games']} | {a['wins']}-{a['draws']}-{a['losses']} | "
            f"{a['win_rate']:.1f} | {a['non_loss_rate']:.1f} | "
            f"{fmt(a['avg_sims'], '.0f')} | "
            f"{fmt(a['avg_cpuct'], '.2f')} | "
            f"{fmt(a['avg_temp'], '.2f')} |"
        )
    lines.append("")
    if report["eval_runs"]:
        lines.append("## Evaluation Runs")
        lines.append("")
        for ev in report["eval_runs"][-5:]:
            lines.append(f"- {ev['timestamp']}: {ev['candidate']} vs {ev['baseline']}, played={ev['played']}/{ev['target_games']}, candidate={ev['candidate_score']} ({ev['candidate_pct']}%), results={ev['results']}, avg_plies={ev['avg_plies']}")
        lines.append("")
    lines.append("## Latest Runtime Counters")
    lines.append("")
    for name in ["InferenceLatencyMs", "InferenceGpuWaitMs", "InferenceBatch", "InferenceQueue", "LegalMovesCacheHit", "LegalMovesCacheMiss", "InferenceCacheHit", "InferenceCacheMiss"]:
        item = report["runtime_counters"].get(name)
        if item:
            lines.append(f"- {name}: samples={item['samples']:,}, avg={item['avg']}, max={item['max']} at {item['timestamp']}")
    lines.append("")
    lines.append("## Errors")
    lines.append("")
    if report["errors"]:
        for msg, count in report["errors"][:10]:
            lines.append(f"- {count}x {msg}")
    else:
        lines.append("- No matched errors.")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Analyze ChineseChessAI runtime.log and extract training/evaluation metrics.")
    parser.add_argument("log", nargs="?", default=r"D:\temp\runtime.log")
    parser.add_argument("--out-json", default=r"D:\temp\runtime_analysis.json")
    parser.add_argument("--out-md", default=r"D:\temp\runtime_analysis.md")
    parser.add_argument("--tail-games", type=int, default=20)
    args = parser.parse_args()

    report = analyze(Path(args.log), args.tail_games)
    json_path = Path(args.out_json)
    md_path = Path(args.out_md)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(json.dumps({
        "range": [report["totals"]["first_timestamp"], report["totals"]["last_timestamp"]],
        "games_completed": report["totals"]["games_completed"],
        "training_runs": report["training"]["runs"],
        "last_loss": report["training"]["last_loss"],
        "draw_rate": report["game_quality"]["draw_rate"],
        "top_agents": report["top_agents"][:5],
        "eval_runs": report["eval_runs"][-3:],
        "top_errors": report["errors"][:3],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
