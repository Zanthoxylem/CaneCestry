# CaneCestry 2.0

*A web‑based toolkit for pedigree exploration and additive/​co‑ancestry matrix computation in breeding programs*

---

## Repository URL
<https://github.com/Zanthoxylem/Canecestry_Pedigrees>

---

## 1  Project structure

```text
Canecestry_Pedigrees/
├── flask_app.py            # Dash + Flask entry point
├── Pedigree_Subset.txt     # Demo pedigree (tab‑delimited)
├── requirements.txt        # Exact Python dependencies (locked)
├── Dockerfile              # Container build recipe
├── docker-compose.yml      # One‑command dev setup
├── README.md               # ← this file
└── tests/                  # Pytest suite (optional)
```

---

## 2  Docker Compose – **recommended**

> **Prerequisite :** Docker Desktop ≥ 4 (Engine ≥ 20.10) on Windows / macOS or Docker Engine on Linux.

```bash
# clone the repository
git clone https://github.com/Zanthoxylem/Canecestry_Pedigrees.git
cd Canecestry_Pedigrees

# build the image and start the container (first run ≈ 4‑5 min)
docker compose up --build
```

Open your browser at **<http://localhost:8050>**.

Stop and clean up:
```bash
docker compose down
```

The **`docker-compose.yml`** mounts the current working directory into the container, so source edits reload when you refresh the browser.

---

## 3  Alternative local install (Python 3.11)

Use this path only if Docker cannot be used.

```bash
git clone https://github.com/Zanthoxylem/Canecestry_Pedigrees.git
cd Canecestry_Pedigrees

python -m venv .venv
source .venv/bin/activate        # Windows → .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python flask_app.py --host 0.0.0.0 --port 8050
```

Browse to **<http://127.0.0.1:8050>**.


***CaneCestry is also avaiable online at https://canecestry-zanthoxylum2117.pythonanywhere.com/***

---

## 4  Key functionality

| Module | Capabilities |
|--------|--------------|
| **Generate Kinship Matrix** | • Compute additive (A) or co‑ancestry matrices via Henderson method. <br>• Heat‑map with hierarchical clustering (SciPy) or fallback simple heat‑map<br>• Download full or user‑defined subset as CSV |
| **Pedigree Explorer** | • Ancestry / descendant tracing, coloured by maternal/paternal lineages<br>• Progeny lookup (single parent or specific cross)<br>• Interactive family‑tree images rendered via Graphviz with kinship colour maps |
| **Add Pedigree Entries** | • Upload tab‑delimited pedigree records<br>• Inline correction of missing/unknown parents<br>• Temporary entries for what‑if analysis |


---

## 5  Built‑in demo dataset

* **File :** `Pedigree_Subset.txt`  
  1 515 genotypes × 3 columns (`LineName`, `FemaleParent`, `MaleParent`).  
  

Users can replace or augment this file with proprietary pedigrees; no code changes are required.

---


## 6  Licensing

* **Software :** MIT Licence (see `LICENSE`).

Users may incorporate CaneCestry into closed‑source pipelines provided the original copyright notice is retained.




Bug reports and feature requests → <https://github.com/Zanthoxylem/Canecestry_Pedigrees/issues>.

