"""Generate LaTeX longtable for Supporting Information from sweep2_results.csv."""
from pathlib import Path

import csv


def fmt(x: str) -> str:
    if not x:
        return "---"
    try:
        v = float(x)
        if abs(v) < 0.0001 and v != 0:
            return f"{v:.2e}"
        return f"{v:.4g}"
    except ValueError:
        return x.replace("_", r"\_")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    csv_path = (
        root / "archive" / "outputs_legacy" / "compass_sweep2" / "sweep" / "sweep2_results.csv"
    )
    out = root / "paper" / "latex_fragments" / "si_sweep_table.tex"
    out.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    header = (
        r"Experiment & $N_{\mathrm{steps}}$ & $T$ (ns) & JSD$_\mathrm{C}$ & JSD$_\mathrm{R}$ & "
        r"$\epsilon_{\mathrm{helix,C}}$ & $\epsilon_{\mathrm{helix,R}}$ \\"
    )

    lines = [
        r"\begin{longtable}{@{}p{0.52\linewidth}rrrrrr@{}}",
        r"\caption{Complete retrospective sweep (enhanced COMPASS, \texttt{sweep2}). "
        r"Subscripts C/R denote COMPASS and bootstrap random mean ($100$ repeats).}\label{tab:si_sweep}\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{7}{c}{\tablename\ \thetable{} --- \textit{continued from previous page}} \\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{7}{r}{\textit{Continued on next page}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    for row in rows:
        exp = row["experiment"].replace("_", r"\_")
        lines.append(
            f"{exp} & {row['n_steps']} & {fmt(row['sampled_ns'])} & "
            f"{fmt(row['compass_jsd_mean'])} & {fmt(row['random_jsd_mean'])} & "
            f"{fmt(row['compass_f_helix_error_mean'])} & {fmt(row['random_f_helix_error_mean'])} \\\\"
        )

    lines.append(r"\end{longtable}")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
