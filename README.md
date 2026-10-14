# PVTOOLS

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Dash](https://img.shields.io/badge/plotly-dash-purple)
![License](https://img.shields.io/badge/license-LBNL-green)
![Status](https://img.shields.io/badge/status-active-success)

PVTools is a collection of **open-source web applications and Python tools for photovoltaic (PV) reliability and system analysis**.

The tools are developed by the **[LBL HackingMaterials Group](https://hackingmaterials.lbl.gov/)** at **Lawrence Berkeley National Laboratory** as part of the **[DuraMAT Consortium](https://www.duramat.org/)**.

PVTools provides interactive tools and datasets to support **PV system design, degradation analysis, and reliability research**.

**Live site: [pvtools.lbl.gov](https://pvtools.lbl.gov)**

---

# Table of Contents

- [Cite our work](#cite-our-work)
- [Applications](#applications)
- [Repository Structure](#repository-structure)
- [Running Locally](#running-locally)
- [Deployment](#deployment)
- [Logging and Privacy](#logging-and-privacy)
- [Contributors](#contributors)
- [Copyright](#copyright)

---

# Cite our work
If you use **PVTOOLS** in your research, please cite:

B. Li, T. Karin, X. Chen, and A. Jain, pvtools: Toolkit for Photovoltaic Data Analysis and Modeling, Zenodo, Jun. 2022. DOI: https://doi.org/10.5281/zenodo.15654456

# Applications

## PV-Copilot

<p align="center">
<img src="assets/pvcopilot_cover.png" width="500">
</p>

Data in, degradation results out: **LLM-assisted automatic analysis of PV field data**.

Features:

- Upload time-series data (or use the example dataset); columns are mapped automatically
- Data loading, filtering, degradation-rate analysis, and PVPRO single-diode parameter extraction
- Simple mode (one click) and Advanced mode (step by step)
- AI code generation, AI diagnostics, and an AI help chat

---

## PV Degradation Pathway

<p align="center">
<img src="assets/pv_pathway_cover.png" width="500">
</p>

Explore **degradation pathways of PV modules** reported in the literature.

---

## PV Image Analysis Using LLMs

<p align="center">
<img src="assets/pv_image_logo.png" width="500">
</p>

Fault detection using **visible, EL, and IR photovoltaic images** with large language models.

Features:

- AI-assisted PV defect detection
- Multi-image modality support
- LLM-assisted interpretation

---

## Global PV Field Performance

<p align="center">
<img src="assets/field_front.jpg" width="500">
</p>

Interactive tool for exploring **global PV degradation datasets**.

Capabilities:

- Field degradation statistics
- Global reliability benchmarking
- Interactive visualization

---

## IV Curve Correction Tool

<p align="center">
<img src="assets/ivcorrection.png" width="500">
</p>

Correct photovoltaic **IV curves** to standard test conditions following:

**IEC 60891:2021**

Features:

- Field IV curve correction
- Comparison between measured and corrected curves
- Export of corrected datasets

---

## String Length Calculator

<p align="center">
<img src="assets/string_length_screenshot.png" width="500">
</p>

Calculate the **maximum allowable PV string length** for a given system location.

Inputs include:

- Module parameters
- Temperature conditions
- Geographic location

The tool follows NEC-compatible PV system design calculations.

---

## Photovoltaic Climate Zones

<p align="center">
<img src="assets/pvcz_screenshow2.jpg" width="500">
</p>

Interactive visualization of **environmental stressors affecting PV systems worldwide**.

Includes:

- Climate zone classification
- Environmental stress distribution
- Reliability stress mapping

---

# Repository Structure

```bash
pvtools/
├── index.py                 # Dash app entry point (navbar, footer, page container)
├── app.py                   # Dash app object and HTML template
├── pages/                   # One file per tool page (pvcopilot.py, field_degradation.py, ...)
├── page_supporting_files/   # Analysis code used by the pages (PV-Copilot pipeline)
├── assets/                  # Images, stylesheets, and client-side JavaScript
├── utils/                   # Shared helpers, incl. usage logging (usage_log.py, usage_db.py)
├── dataset_review/          # Internal dataset review app (mounted at /dataset-review/)
├── loadtest/                # Playwright load-test script
├── Dockerfile / Procfile    # Production container (gunicorn)
└── requirements.txt         # Python dependencies
```

# Running Locally

```bash
git clone https://github.com/lbj2011/pvtools.git
cd pvtools
pip install -r requirements.txt
python index.py              # then open http://127.0.0.1:8050
```

API keys for the LLM features (e.g. `OPENAI_API_KEY`) are read from environment variables or a local `.env` file, which is not committed.

# Deployment

pvtools.lbl.gov runs on Heroku as a Docker container:

- `gunicorn index:server -k gthread --workers 2 --threads 8 --timeout 600`
- Long-running PV-Copilot jobs share state between workers through `diskcache`
- Usage logs are stored in Heroku Postgres (`DATABASE_URL`), so they survive deploys and restarts

# Logging and Privacy

We take privacy seriously.

We log basic usage information (page visits, which buttons are used, and how long each analysis step takes) to understand how the tools are used and to improve performance. IP addresses are stored only as hashes; the approximate country comes from the request headers.

We do not store the contents of uploaded data files. Uploaded data is kept only temporarily on the server while the analysis runs.


# Contributors
Baojie Li, Todd Karin, Xin Chen, Anubhav Jain


# Copyright

© 2026 PVTOOLS | Lawrence Berkeley National Laboratory

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative 
works, and perform publicly and display publicly, and to permit others to do so.
