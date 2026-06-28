from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from vitrail.atelier.journal import get_logger
from vitrail.atelier.provenance import read_pane
from vitrail.atelier.schema import Runtime, TaskSpec, TopConfig, load_config
from vitrail.bench.finetune import run_finetune
from vitrail.bench.pretrain import run_pretrain
from vitrail.lights.panel import Window
from vitrail.studio.evaluate import appraise
from vitrail.studio.onnx import export_encoder

_LOG = get_logger("studio")


def _overrides(pairs: list[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for item in pairs:
        key, _, raw = item.partition("=")
        out[key] = yaml.safe_load(raw)
    return out


def _runtime(args: argparse.Namespace) -> Runtime:
    rt = Runtime()
    if args.device:
        rt = rt.model_copy(update={"device": args.device})
    if args.out_dir:
        rt = rt.model_copy(update={"out_dir": Path(args.out_dir)})
    return rt


def _config(args: argparse.Namespace) -> TopConfig:
    return load_config(args.config, _overrides(args.set))


def _task_spec(config: TopConfig, name: str) -> TaskSpec:
    if config.task is not None and config.task.name == name:
        return config.task
    return TaskSpec(name=name)


def _cmd_pretrain(args: argparse.Namespace) -> None:
    config = _config(args)
    run_pretrain(config, _runtime(args), corpus_len=args.corpus_len, max_steps=args.max_steps)


def _cmd_finetune(args: argparse.Namespace) -> None:
    config = _config(args)
    task = _task_spec(config, args.task)
    run_finetune(config, _runtime(args), task, pretrained=args.pretrained, max_steps=args.max_steps)


def _cmd_appraise(args: argparse.Namespace) -> None:
    config = _config(args)
    runtime = _runtime(args)
    window = Window(config.cartoon, config.glass, config.regimen).to(runtime.device)
    if args.pane:
        window.load_state_dict(read_pane(args.pane)["model"], strict=False)
    scores = appraise(window, config, runtime, args.task)
    for key, value in scores.items():
        _LOG.info("%s = %.4f", key, value)


def _cmd_etch(args: argparse.Namespace) -> None:
    config = _config(args)
    window = Window(config.cartoon, config.glass, config.regimen)
    if args.pane:
        window.load_state_dict(read_pane(args.pane)["model"], strict=False)
    path = export_encoder(window.encoder, config.glass.patch, args.out)
    _LOG.info("etched %s", path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vitrail")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    pre = sub.add_parser("pretrain")
    pre.add_argument("--config", required=True)
    pre.add_argument("--set", nargs="*", default=[])
    pre.add_argument("--corpus-len", type=int, default=64)
    pre.add_argument("--max-steps", type=int, default=None)
    pre.set_defaults(func=_cmd_pretrain)

    fit = sub.add_parser("finetune")
    fit.add_argument("--config", required=True)
    fit.add_argument("--task", required=True, choices=["seg_cbct", "seg_ios", "fusion", "implant"])
    fit.add_argument("--pretrained", default=None)
    fit.add_argument("--set", nargs="*", default=[])
    fit.add_argument("--max-steps", type=int, default=None)
    fit.set_defaults(func=_cmd_finetune)

    ap = sub.add_parser("appraise")
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", required=True, choices=["seg_cbct", "seg_ios", "fusion", "implant"])
    ap.add_argument("--pane", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.set_defaults(func=_cmd_appraise)

    etch = sub.add_parser("etch")
    etch.add_argument("--config", required=True)
    etch.add_argument("--out", required=True)
    etch.add_argument("--pane", default=None)
    etch.add_argument("--set", nargs="*", default=[])
    etch.set_defaults(func=_cmd_etch)
    return parser


def main(argv: list[str] | None = None) -> None:
    torch.set_num_threads(max(torch.get_num_threads(), 1))
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
