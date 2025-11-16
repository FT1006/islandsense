# IslandSense

Per-sailing disruption prediction and 7-day early warning system for Jersey supply chain resilience.

## Overview

IslandSense predicts cancellation risk for ferry sailings to Jersey and provides actionable recommendations to minimize supply chain disruption for critical goods (fresh produce, fuel).

### Key Features

- **Per-Sailing Risk Scores**: ML-based cancellation probability for each scheduled sailing
- **7-Day Risk Dashboard**: Weekly outlook with daily risk bands (Low/Moderate/High)
- **Scenario Planning**: Compare mitigation strategies (forward shipping, air-lift contingencies)
- **Feature Transparency**: View risk drivers (wind, tide, beam-sea exposure) for each sailing

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Synthetic  │ ──▶ │   ML Model   │ ──▶ │  Aggregate  │
│    Data     │     │  (XGBoost)   │     │   Pipeline  │
└─────────────┘     └──────────────┘     └─────────────┘
                                                │
                                                ▼
                    ┌──────────────┐     ┌─────────────┐
                    │   React UI   │ ◀── │  FastAPI    │
                    │  Dashboard   │     │   Server    │
                    └──────────────┘     └─────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- npm

### Installation

```bash
# Install Python dependencies
pip install -e .
pip install fastapi uvicorn

# Install frontend dependencies
cd frontend
npm install
```

### Run Locally

**Backend API:**
```bash
uvicorn src.islandsense.api:app --host 0.0.0.0 --port 8000
```

**Frontend (development):**
```bash
cd frontend
npm run dev
```

Access at http://localhost:5173

### Production Build

```bash
cd frontend
npm run build
```

Then the FastAPI server serves both API and static frontend.

## Deployment

### Railway (Recommended)

1. Push to GitHub
2. Connect repo to Railway
3. Auto-deploys with Dockerfile

### Docker

```bash
docker build -t islandsense .
docker run -p 8000:8000 islandsense
```

## Project Structure

```
islandsense/
├── src/islandsense/     # Core Python package
│   ├── api.py           # FastAPI server
│   ├── aggregate.py     # Risk aggregation pipeline
│   ├── model.py         # ML model training
│   ├── features.py      # Feature engineering
│   └── config.py        # Configuration loader
├── frontend/            # React + TypeScript UI
│   ├── src/App.tsx      # Main dashboard component
│   └── dist/            # Production build
├── data/                # CSV datasets
│   ├── sailing_contrib.csv
│   ├── daily_risk.csv
│   ├── weekly_risk.csv
│   └── scenario_impact.csv
├── config.yaml          # Risk thresholds and scenarios
├── Dockerfile           # Container configuration
└── railway.json         # Railway deployment config
```

## Risk Scoring

- **0-20**: Low Risk (Green)
- **21-50**: Moderate Risk (Amber)
- **51-100**: High Risk (Red)

Weekly score is the average of daily risk scores.

## Scenarios

- **Optimized**: Forward ship 10% of cargo on lower-risk sailings
- **Aggressive**: Forward 10% + air-lift 5% contingency for critical items

## Tech Stack

- **Backend**: Python, FastAPI, Pandas, scikit-learn
- **Frontend**: React, TypeScript, Tailwind CSS, shadcn/ui
- **ML**: XGBoost with isotonic calibration
- **Deployment**: Docker, Railway

## License

MIT
