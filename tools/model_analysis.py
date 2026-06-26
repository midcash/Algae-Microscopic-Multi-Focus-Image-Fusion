"""模型参数量统计工具 — 支持YAML配置, JSON输出, 未来可扩展柱状图
用法:
  python tools/model_analysis.py                          # 使用默认YAML配置
  python tools/model_analysis.py --config analysis_cfg.yaml
  python tools/model_analysis.py --module m_segnet_v5     # 单模型快速统计
"""
import sys, os, argparse, json, importlib
from pathlib import Path
from datetime import datetime
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ===== 默认配置文件 =====
DEFAULT_YAML = """
# 参数量对比配置文件
# 每个entry必须: module(模型模块路径), model_class(类名), create_fn(工厂函数,可选)
# 可选: kwargs(构造参数), label(显示名称), group_rules(自定义分组)

entries:
  - label: "V5 (Ours)"
    module: "models.m_segnet_v5"
    model_class: "MSegNetV5"
    kwargs:
      num_source_images: 5
      use_fusion_head: gumbel
      top_k: 1
      stem_channels: 24
      stage_channels: [24, 48, 96, 128]
      stage_blocks: [2, 4, 6, 3]
      bifpn_out_channels: 64
      bifpn_num_layers: 2
      decoder_tail_channels: 8
      cross_source_alpha: 0.1

  - label: "m-SegNet (V1 prototype)"
    module: "models.m_segnet"
    model_class: "MSegNet"
    kwargs:
      num_source_images: 5
      use_fusion_head: gumbel
"""

# ===== 默认分组规则 =====
DEFAULT_GROUPS = {
    "Encoder":    ["encoder"],
    "SPPF":       ["sppf"],
    "BiFPN":      ["bifpn"],
    "Decoder":    ["decoder"],
    "DecisionNet":["decision_net", "fusion_head.decision_net"],
    "FusionHead": ["fusion_head", "!fusion_head.decision_net"],
    "SimAM":      ["simam"],
}


def load_yaml_config(path: str) -> dict:
    """加载YAML配置文件, 如果没有yaml库则使用内建parser"""
    try:
        import yaml
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except ImportError:
        # Fallback: 简单YAML parser (支持基本格式)
        return _simple_yaml_parse(path)


def _simple_yaml_parse(path: str) -> dict:
    """简易YAML解析, 仅支持本工具所需的基本格式"""
    with open(path, 'r') as f:
        content = f.read()
    # 对于复杂配置, 建议 pip install pyyaml
    raise ImportError("Please install pyyaml: pip install pyyaml")


def count_params_by_prefix(model, prefixes, exclude_prefixes=None):
    """统计模型中以指定前缀开头的模块的参数量"""
    total = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        matched = any(name.startswith(p) for p in prefixes)
        if exclude_prefixes:
            matched = matched and not any(name.startswith(p) for p in exclude_prefixes)
        if matched:
            total += param.numel()
    return total


def analyze_model(model, label, group_rules=None):
    """分析单个模型的参数量"""
    if group_rules is None:
        group_rules = DEFAULT_GROUPS

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 按分组规则统计
    groups = {}
    allocated = 0
    for group_name, prefixes in group_rules.items():
        include = [p for p in prefixes if not p.startswith("!")]
        exclude = [p[1:] for p in prefixes if p.startswith("!")]
        count = count_params_by_prefix(model, include, exclude)
        if count > 0:
            groups[group_name] = count
            allocated += count

    # 未归类参数
    other = total - allocated
    if other > 0:
        groups["Other"] = other

    return {
        "label": label,
        "total_params": total,
        "groups": groups,
    }


def print_results(results):
    """打印结果表格"""
    print(f"\n{'='*60}")
    print(f"  Model Parameter Analysis")
    print(f"{'='*60}")

    for r in results:
        print(f"\n--- {r['label']} ---")
        print(f"  Total: {r['total_params']:,} ({r['total_params']/1e6:.2f}M)")
        for gname, gcount in r['groups'].items():
            pct = gcount / r['total_params'] * 100 if r['total_params'] > 0 else 0
            print(f"    {gname:20s}: {gcount:>10,} ({pct:5.1f}%)")

    # 对比表
    if len(results) >= 2:
        print(f"\n{'='*60}")
        print(f"  Comparison")
        print(f"{'='*60}")
        all_groups = sorted(set(g for r in results for g in r['groups']))
        header = f"{'Module':20s}"
        for r in results:
            header += f" {r['label']:>18s}"
        print(header)
        print("-" * len(header))
        for g in all_groups:
            row = f"{g:20s}"
            for r in results:
                val = r['groups'].get(g, 0)
                row += f" {val:>18,}"
            print(row)
        # Total row
        row = f"{'TOTAL':20s}"
        for r in results:
            row += f" {r['total_params']:>18,}"
        print("-" * len(header))
        print(row)


def save_json(results, outpath):
    """保存为JSON"""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "results": [{k: v for k, v in r.items()} for r in results],
    }
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {outpath}")


def main():
    ap = argparse.ArgumentParser(description="Model Parameter Analysis Tool")
    ap.add_argument("--config", type=str, default=None, help="YAML config file path")
    ap.add_argument("--module", type=str, default=None, help="Quick: single model module path")
    ap.add_argument("--class", dest="cls", type=str, default=None, help="Quick: model class name")
    ap.add_argument("--kwargs", type=str, default="{}", help="Quick: kwargs as JSON string")
    ap.add_argument("--output", type=str, default="output/model_analysis.json", help="JSON output path")
    args = ap.parse_args()

    results = []

    if args.config:
        cfg = load_yaml_config(args.config)
        for entry in cfg.get("entries", []):
            mod = importlib.import_module(entry["module"])
            cls = getattr(mod, entry["model_class"])
            kw = entry.get("kwargs", {})
            label = entry.get("label", entry["model_class"])
            model = cls(**kw)
            results.append(analyze_model(model, label, entry.get("group_rules")))
    elif args.module:
        mod = importlib.import_module(args.module)
        cls = getattr(mod, args.cls or "create_model")
        kw = json.loads(args.kwargs) if args.kwargs else {}
        if hasattr(cls, '__call__') and args.cls != "create_model":
            model = cls(**kw)
        else:
            model = cls(**kw) if kw else cls()
        label = args.cls or args.module
        results.append(analyze_model(model, label))
    else:
        # Default: 使用内建DEFAULT_YAML
        # Default: 使用内建DEFAULT_YAML
        try:
            import yaml
            cfg = yaml.safe_load(DEFAULT_YAML)
            for entry in cfg.get("entries", []):
                mod = importlib.import_module(entry["module"])
                cls = getattr(mod, entry["model_class"])
                kw = entry.get("kwargs", {})
                label = entry.get("label", entry["model_class"])
                model = cls(**kw)
                results.append(analyze_model(model, label, entry.get("group_rules")))
        except ImportError:
            from models.m_segnet_v5 import create_model as create_v5
            from models.m_segnet import create_model as create_v1
            results.append(analyze_model(
                create_v5(num_source_images=5, use_fusion_head='gumbel', top_k=1), "V5 (Ours)"))
            results.append(analyze_model(
                create_v1(num_source_images=5, use_fusion_head='gumbel'), "m-SegNet V1"))

    if results:
        print_results(results)
        save_json(results, args.output)
    else:
        print("No models to analyze. Use --config or --module.")


if __name__ == "__main__":
    main()
