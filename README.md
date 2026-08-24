# Safe Blower Coordination — Open Demonstration

This repository is an **open demonstration** of the method described in:

> Cirillo, L.; Gotelli, M.; Massei, M.; Sina, X.; Solina, V.  
> *A Simulation-Based Framework for Energy-Efficient and Safe Blower Coordination in Wastewater Treatment Plants.*  
> Energies **2025**, 18, 5947.  
> https://doi.org/10.3390/en18225947

It shows the published modelling and optimization ideas (blower map, network hydraulics, grid search, and Safe Bayesian Optimization) in a form that can be run without plant access.

## What this is — and what it is not

This is **not** the complete software used with the industrial partner.

The full application cannot be published. It contains **client operational data**, site-specific configuration, and vendor performance information that we are not allowed to release. That includes SCADA/PLC historian records, measured flows, pressures and power, and manufacturer performance-data-sheet (PDS) tables.

This repository is therefore a **privacy-preserving demo**:

| Included | Not included |
|---|---|
| Method from the paper (blower model, header hydraulics, GS vs SBO) | Client wastewater-plant measurements |
| Public Table 1 parameters (3 × 78 kW blowers, 51 kPa header, 4600 Nm³ rating) | Vendor PDS curves and serial-numbered equipment data |
| Synthetic map and demand scenarios, clearly labelled as stand-ins | Site piping geometry, valve schedules, operator set-points |
| Scripts to reproduce a GS vs SBO comparison | Production GUI, SCADA interface, or deployment tooling |

Results here illustrate the **same control problem and algorithms**. They are **not** a bit-for-bit replica of Table 2 in the paper, because that table was produced with unpublished vendor maps and the restricted plant model.

## How to run

Python 3.10+ recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python compare_gs_sbo.py
```

One optimizer at a time:

```bash
python main.py -o grid
python main.py -o sbo
```

Outputs are printed to the terminal. `compare_gs_sbo.py` also writes `comparison_results.json`.

## Citation

If you use this demo, please cite the Energies paper (DOI above).

The code layout was produced with the help of [PaperCoder / Paper2Code](https://github.com/going-doer/Paper2Code) (Seo et al., ICLR 2026) from the public article text, then calibrated so the demonstration actually executes without confidential inputs.

## Contact

Questions about the **scientific method**: see the paper and the corresponding author listed there.  
Questions about **plant data or the production tool**: those materials stay with the project partners and are available only under the agreements that cover them.
