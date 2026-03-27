#!/usr/bin/env python3
"""
hardy_weinberg_simulator.py — Wright-Fisher HWE Departure Simulator
Eric D. Martin — All Your Baseline LLC
2026-03-27

Treats Hardy-Weinberg Equilibrium as a zero-drift baseline, then simulates
violations to find the drift threshold — the point at which a real population
departs consistently from HWE expectations.

Goal: turn HWE from a static ideal into a testable benchmark.

Usage:
    python3 hardy_weinberg_simulator.py              # run all scenarios
    python3 hardy_weinberg_simulator.py --sweep      # parameter sweep
    python3 hardy_weinberg_simulator.py --plot       # generate figures
"""

import numpy as np
import argparse
import sys

# ── HWE core ──────────────────────────────────────────────────────────────────

def expected_genotype_freqs(p):
    """Given allele freq p, return HWE expected (AA, Aa, aa) frequencies."""
    q = 1.0 - p
    return p**2, 2*p*q, q**2


def chi2_hwe_test(n_AA, n_Aa, n_aa):
    """
    Chi-squared test for HWE departure.
    Returns (chi2_stat, p_value, D_hwe).

    D_hwe = observed heterozygosity - expected heterozygosity
    (positive = excess heterozygotes; negative = deficit)
    """
    from scipy.stats import chi2 as chi2_dist

    N = n_AA + n_Aa + n_aa
    if N == 0:
        return 0.0, 1.0, 0.0

    p_obs = (2 * n_AA + n_Aa) / (2 * N)
    q_obs = 1.0 - p_obs

    exp_AA = p_obs**2 * N
    exp_Aa = 2 * p_obs * q_obs * N
    exp_aa = q_obs**2 * N

    # Avoid division by zero
    def term(obs, exp):
        return (obs - exp)**2 / exp if exp > 0 else 0.0

    chi2_stat = term(n_AA, exp_AA) + term(n_Aa, exp_Aa) + term(n_aa, exp_aa)
    p_value = 1.0 - chi2_dist.cdf(chi2_stat, df=1)  # 1 df for HWE test

    # HWE departure index: positive = excess het
    D_hwe = (n_Aa / N) - (2 * p_obs * q_obs)

    return chi2_stat, p_value, D_hwe


# ── Wright-Fisher simulator ────────────────────────────────────────────────────

def wright_fisher_step(counts, N, p, selection_s=0.0, mutation_u=0.0,
                       inbreeding_F=0.0, migration_m=0.0, migration_p=0.5):
    """
    One generation of Wright-Fisher evolution.

    Parameters
    ----------
    counts : (n_AA, n_Aa, n_aa)
    N      : population size
    p      : current allele frequency of A
    selection_s  : selection coefficient (positive = A favored)
    mutation_u   : mutation rate A→a (per generation)
    inbreeding_F : inbreeding coefficient (0=random mating, 1=full sib)
    migration_m  : migration rate (fraction replaced by migrants each gen)
    migration_p  : allele frequency in migrant pool

    Returns
    -------
    (n_AA, n_Aa, n_aa), new_p
    """
    q = 1.0 - p

    # Selection: adjust allele frequencies
    if selection_s != 0.0:
        w_AA = 1.0 + selection_s
        w_Aa = 1.0 + 0.5 * selection_s  # additive
        w_aa = 1.0
        w_bar = p**2 * w_AA + 2*p*q * w_Aa + q**2 * w_aa
        p_sel = (p**2 * w_AA + p*q * w_Aa) / w_bar
    else:
        p_sel = p

    # Mutation: A → a
    if mutation_u > 0.0:
        p_mut = p_sel * (1.0 - mutation_u)
    else:
        p_mut = p_sel

    # Migration: replace fraction m with migrants
    if migration_m > 0.0:
        p_mig = (1.0 - migration_m) * p_mut + migration_m * migration_p
    else:
        p_mig = p_mut

    # Inbreeding: genotype frequencies with inbreeding F
    p_f = p_mig
    q_f = 1.0 - p_f
    freq_AA = p_f**2 * (1 - inbreeding_F) + p_f * inbreeding_F
    freq_Aa = 2 * p_f * q_f * (1 - inbreeding_F)
    freq_aa = q_f**2 * (1 - inbreeding_F) + q_f * inbreeding_F

    # Normalize (floating point safety)
    total = freq_AA + freq_Aa + freq_aa
    freq_AA /= total
    freq_Aa /= total
    freq_aa /= total

    # Drift: multinomial sampling
    genotype_counts = np.random.multinomial(N, [freq_AA, freq_Aa, freq_aa])
    n_AA, n_Aa, n_aa = genotype_counts

    # New allele frequency from genotype counts
    new_p = (2 * n_AA + n_Aa) / (2 * N)

    return (int(n_AA), int(n_Aa), int(n_aa)), new_p


def simulate_population(N=500, p0=0.5, n_generations=100,
                        selection_s=0.0, mutation_u=0.0,
                        inbreeding_F=0.0, migration_m=0.0, migration_p=0.5,
                        hwe_alpha=0.05):
    """
    Simulate one population trajectory and detect HWE departure.

    Returns
    -------
    dict with:
        p_history       : allele freq per generation
        chi2_history    : HWE chi2 per generation
        D_hwe_history   : HWE departure index per generation
        drift_gen       : first generation of sustained HWE departure (or None)
        final_p         : final allele frequency
    """
    p = p0
    q = 1.0 - p
    n_AA = int(round(p**2 * N))
    n_aa = int(round(q**2 * N))
    n_Aa = N - n_AA - n_aa

    p_history = [p]
    chi2_history = []
    D_hwe_history = []
    departure_streak = 0
    drift_gen = None
    STREAK_REQUIRED = 3  # consecutive generations below alpha to call drift

    for gen in range(n_generations):
        (n_AA, n_Aa, n_aa), p = wright_fisher_step(
            (n_AA, n_Aa, n_aa), N, p,
            selection_s=selection_s,
            mutation_u=mutation_u,
            inbreeding_F=inbreeding_F,
            migration_m=migration_m,
            migration_p=migration_p
        )
        chi2_stat, p_val, D_hwe = chi2_hwe_test(n_AA, n_Aa, n_aa)

        p_history.append(p)
        chi2_history.append(chi2_stat)
        D_hwe_history.append(D_hwe)

        if p_val < hwe_alpha:
            departure_streak += 1
            if departure_streak >= STREAK_REQUIRED and drift_gen is None:
                drift_gen = gen + 1
        else:
            departure_streak = 0

        # Fixation check
        if p <= 0.0 or p >= 1.0:
            break

    return {
        "p_history": np.array(p_history),
        "chi2_history": np.array(chi2_history),
        "D_hwe_history": np.array(D_hwe_history),
        "drift_gen": drift_gen,
        "final_p": p,
        "n_generations_run": len(p_history) - 1,
    }


# ── Parameter sweep ────────────────────────────────────────────────────────────

def run_sweep(n_reps=200):
    """
    Sweep N, p0, selection, inbreeding across parameter space.
    Reports mean drift point and fixation probability.
    """
    scenarios = [
        # (label, N, p0, s, F, mu, m)
        ("Ideal HWE",           500,  0.5,  0.0,  0.0, 0.0, 0.0),
        ("Small N (N=50)",       50,  0.5,  0.0,  0.0, 0.0, 0.0),
        ("Small N (N=20)",       20,  0.5,  0.0,  0.0, 0.0, 0.0),
        ("Rare allele p=0.1",   500,  0.1,  0.0,  0.0, 0.0, 0.0),
        ("Selection s=0.05",    500,  0.5,  0.05, 0.0, 0.0, 0.0),
        ("Selection s=0.1",     500,  0.5,  0.1,  0.0, 0.0, 0.0),
        ("Inbreeding F=0.1",    500,  0.5,  0.0,  0.1, 0.0, 0.0),
        ("Inbreeding F=0.3",    500,  0.5,  0.0,  0.3, 0.0, 0.0),
        ("Mutation u=0.01",     500,  0.5,  0.0,  0.0, 0.01,0.0),
        ("Migration m=0.05",    500,  0.5,  0.0,  0.0, 0.0, 0.05),
        ("Combined (s+F)",      500,  0.5,  0.05, 0.1, 0.0, 0.0),
    ]

    print(f"\n{'Scenario':<28} {'MeanDrift':>10} {'%Departed':>10} {'%Fixed':>8} {'FinalP_μ':>9}")
    print("-" * 68)

    results = {}
    for label, N, p0, s, F, mu, m in scenarios:
        drift_gens = []
        departed = 0
        fixed = 0
        final_ps = []

        for _ in range(n_reps):
            res = simulate_population(
                N=N, p0=p0, n_generations=200,
                selection_s=s, inbreeding_F=F, mutation_u=mu, migration_m=m
            )
            if res["drift_gen"] is not None:
                drift_gens.append(res["drift_gen"])
                departed += 1
            if res["final_p"] <= 0.01 or res["final_p"] >= 0.99:
                fixed += 1
            final_ps.append(res["final_p"])

        mean_drift = np.mean(drift_gens) if drift_gens else float("nan")
        pct_departed = 100 * departed / n_reps
        pct_fixed = 100 * fixed / n_reps
        mean_final_p = np.mean(final_ps)

        print(f"{label:<28} {mean_drift:>10.1f} {pct_departed:>10.1f} {pct_fixed:>8.1f} {mean_final_p:>9.3f}")
        results[label] = {
            "mean_drift_gen": mean_drift,
            "pct_departed": pct_departed,
            "pct_fixed": pct_fixed,
            "mean_final_p": mean_final_p,
        }

    return results


# ── Single run demo ────────────────────────────────────────────────────────────

def run_demo():
    print("Hardy-Weinberg Drift Simulator — Demo Run")
    print("=" * 55)

    configs = [
        ("Ideal HWE (N=500, p=0.5)",   dict(N=500,  p0=0.5, selection_s=0.0,  inbreeding_F=0.0)),
        ("Small pop (N=50)",            dict(N=50,   p0=0.5, selection_s=0.0,  inbreeding_F=0.0)),
        ("Selection s=0.1",             dict(N=500,  p0=0.5, selection_s=0.1,  inbreeding_F=0.0)),
        ("Inbreeding F=0.2",            dict(N=500,  p0=0.5, selection_s=0.0,  inbreeding_F=0.2)),
    ]

    for label, kwargs in configs:
        res = simulate_population(n_generations=100, **kwargs)
        drift = res["drift_gen"]
        final_p = res["final_p"]
        n_gen = res["n_generations_run"]
        print(f"\n{label}")
        print(f"  Generations run : {n_gen}")
        print(f"  Final p(A)      : {final_p:.4f}")
        print(f"  HWE drift point : {'Gen ' + str(drift) if drift else 'No sustained departure'}")
        if len(res["D_hwe_history"]) > 0:
            print(f"  Max |D_hwe|     : {np.max(np.abs(res['D_hwe_history'])):.4f}")


# ── Optional plotting ──────────────────────────────────────────────────────────

def run_plots():
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        mpl.rcParams.update({"font.size": 11, "font.family": "serif"})
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Wright-Fisher HWE Departure Simulator", fontsize=13)

    configs = [
        ("Ideal HWE (N=500)",    dict(N=500,  p0=0.5, selection_s=0.0,  inbreeding_F=0.0), "steelblue"),
        ("Small pop (N=50)",     dict(N=50,   p0=0.5, selection_s=0.0,  inbreeding_F=0.0), "darkorange"),
        ("Selection s=0.1",      dict(N=500,  p0=0.5, selection_s=0.1,  inbreeding_F=0.0), "crimson"),
        ("Inbreeding F=0.2",     dict(N=500,  p0=0.5, selection_s=0.0,  inbreeding_F=0.2), "purple"),
    ]

    for ax, (label, kwargs, color) in zip(axes.flat, configs):
        res = simulate_population(n_generations=150, **kwargs)
        gens = np.arange(len(res["p_history"]))

        ax.plot(gens, res["p_history"], color=color, lw=1.8, label="p(A)")
        ax.axhline(kwargs["p0"], color="gray", ls="--", lw=1, alpha=0.6, label="p₀")

        if res["drift_gen"] is not None:
            ax.axvline(res["drift_gen"], color="red", ls=":", lw=1.5,
                       label=f"Drift gen={res['drift_gen']}")

        ax.set_title(label)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Allele freq p(A)")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    outfile = "/scratch/repos/ncf-research/hwe_simulator_demo.png"
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {outfile}")
    plt.close()


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hardy-Weinberg Drift Simulator")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--plot",  action="store_true", help="Generate trajectory plots")
    parser.add_argument("--reps",  type=int, default=200, help="Reps per scenario in sweep")
    args = parser.parse_args()

    if args.sweep:
        run_sweep(n_reps=args.reps)
    elif args.plot:
        run_demo()
        run_plots()
    else:
        run_demo()
        print("\nRun with --sweep for full parameter sweep, --plot for figures.")
