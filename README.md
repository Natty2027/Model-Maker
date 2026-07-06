# DS852 Model Lab

A spreadsheet-modeling study companion for **DS852 — Managerial Decision Making**
(Winston & Albright, *Practical Management Science*). It teaches the modeling
framework, the Excel formulas behind it, and how to build and test optimization
and simulation models — then has you build real models in your own Excel.

Built with React + Vite + Tailwind. Deploys to GitHub Pages with one click of
"enable Pages."

## What's inside

| Tab | What it does |
| --- | --- |
| **Framework** | The four model-element types (inputs / decisions / uncertain / output), the build process, and spreadsheet-engineering discipline. |
| **Formulas** | 36 searchable function cards — syntax, *how*, *why*, a worked example, and the pitfall to avoid. Covers SUMPRODUCT, VLOOKUP, INDEX/MATCH, PMT, NPV, RAND/NORM.INV, and more. |
| **Show Me How** | Step-by-step Excel walkthroughs (Solver, data tables, Goal Seek, Monte Carlo, range names, auditing) with copyable formulas. |
| **Build & Solve** | A live two-phase-simplex + branch-and-bound solver. Enter a model, solve it, and get the exact Excel Solver setup to reproduce it. |
| **Test & Practice** | Cholette's model-validation checks run live, plus an 18-question concept quiz. |
| **Randy Example** | The newsvendor problem worked end to end with a live sensitivity chart. |
| **Project Mode** | Guided, stage-by-stage builds in your own Excel with checkpoints verified against the solver. |

## Quick start (run it locally)

```bash
npm install
npm run dev
```

Then open the URL Vite prints (usually http://localhost:5173).

To make a production build:

```bash
npm run build      # outputs to dist/
npm run preview    # serve the built site locally
```

## Deploy to GitHub Pages

1. Create a new GitHub repository and push this project to the **`main`** branch:

   ```bash
   git init
   git add .
   git commit -m "DS852 Model Lab"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```

2. In the repo, go to **Settings ▸ Pages ▸ Build and deployment**, and set
   **Source** to **GitHub Actions**.

3. That's it. The included workflow (`.github/workflows/deploy.yml`) builds the
   site and publishes it on every push to `main`. Your site appears at
   `https://<your-username>.github.io/<your-repo>/`.

`vite.config.js` uses `base: "./"`, so the site works at that project URL
without hard-coding the repo name.

## Tech stack

- React 18
- Vite 5
- Tailwind CSS 3
- Recharts (sensitivity charts)
- lucide-react (icons)

The optimization engine is a self-contained two-phase primal simplex with
branch-and-bound for integer variables — no external solver, no server. It's
unit-tested against classic problems (product mix, transportation, integer
staffing) and detects unbounded and infeasible models.

## A note on academic use

This is a **study companion**: clarify concepts, check formula syntax, and
practice building and testing models. It is designed to help you understand the
material — not to complete graded labs or generate submissions for you. Use it
in line with the DS852 AI policy.

## License

MIT — see [LICENSE](./LICENSE).
