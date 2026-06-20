import argparse
import os
import re

try:
    
    import matplotlib.pyplot as plt
    
except ModuleNotFoundError:
    
    plt = None


def _read_text(path):
    
    if not os.path.exists(path):
        
        return ""
    
    with open(path) as f:
        
        return f.read()


def _first_float(pattern, text):
    
    match = re.search(pattern, text)
    
    if not match:
        
        return None
    
    return float(match.group(1))


def _fmt(value):
    
    if value is None:
        
        return "N/A"
    
    return f"{value:.3f}"


def collect_summary_metrics(testing_dir):
    
    rls_text = _read_text(os.path.join(testing_dir, "rls.txt"))
    ss_text = _read_text(os.path.join(testing_dir, "ss.txt"))
    ade_text = _read_text(os.path.join(testing_dir, "ade_fde_k.txt"))
    coverage_text = _read_text(os.path.join(testing_dir, "reliability", "coverage_summary.txt"))
    
    return {
        "R_avg (%)": _first_float(r"RLS:\s*avg:\s*([0-9.]+)", rls_text),
        "R_min (%)": _first_float(r"min:\s*([0-9.]+)\s*%", rls_text),
        "Coverage_68 (%)": _first_float(r"Coverage_68:\s*([0-9.]+)", coverage_text),
        "Coverage_95 (%)": _first_float(r"Coverage_95:\s*([0-9.]+)", coverage_text),
        "S_68 avg volume": _first_float(r"SS @ 0\.68 %:\s*([0-9.]+)", ss_text),
        "S_95 avg volume": _first_float(r"SS @ 0\.95 %:\s*([0-9.]+)", ss_text),
        "ADE": _first_float(r"min ADE:\s*([0-9.]+)", ade_text),
        "FDE": _first_float(r"min FDE:\s*([0-9.]+)", ade_text)
    }


def generate_summary_table(testing_dir, model_name="LSTM-MDN", title="LSTM-MDN Evaluation Summary"):
    
    metrics = collect_summary_metrics(testing_dir=testing_dir)
    output_path = os.path.join(testing_dir, "evaluation_summary.png")
    
    if plt is None:
        
        with open(os.path.join(testing_dir, "evaluation_summary.txt"), "w") as f:
            
            for metric, value in metrics.items():
                
                f.write(f"{metric}: {_fmt(value)}\n")
        
        return output_path
    
    rows = [[metric, _fmt(value)] for metric, value in metrics.items()]
    
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    ax.axis("off")
    ax.set_title(title, fontsize=22, pad=22)
    
    table = ax.table(
        cellText=rows,
        colLabels=["Metric", model_name],
        cellLoc="center",
        colLoc="center",
        loc="center"
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(17)
    table.scale(1.35, 1.9)
    
    for (_, _), cell in table.get_celld().items():
        
        cell.set_linewidth(1.5)
        cell.set_edgecolor("black")
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    return output_path


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--testing-dir", required=True)
    parser.add_argument("--model-name", default="LSTM-MDN")
    parser.add_argument("--title", default="LSTM-MDN Evaluation Summary")
    args = parser.parse_args()
    
    path = generate_summary_table(
        testing_dir=args.testing_dir,
        model_name=args.model_name,
        title=args.title
    )
    print(path)
    
    return


if __name__ == "__main__":
    
    main()
