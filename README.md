# Diffusion Monte Carlo Streamlit App

Interactive Streamlit application for experimenting with a 3D diffusion Monte Carlo (DMC) simulation in a harmonic potential.

## Features

- configurable number of walkers, time steps, and step size
- optional branching with adaptive reference energy `E_T`
- interactive 3D walker visualization
- time series for reference energy, walker population, radius, and potential
- export of simulation statistics as CSV

## Local start

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

## Files

- `app.py`: Streamlit interface
- `diffusion.py`: diffusion step for walkers
- `branching.py`: branching and population control
- `potential.py`: harmonic potential

## Deploy to Streamlit Community Cloud

1. Push this project to a public GitHub repository.
2. Open [share.streamlit.io](https://share.streamlit.io/).
3. Click `Create app`.
4. Select your repository and choose `app.py` as the entrypoint.
5. Choose a memorable public URL, for example `dmc-simulator.streamlit.app`.
6. Deploy the app and copy the public link into your CV.

## CV text idea

`Interactive Diffusion Monte Carlo simulator (Streamlit): public web app for visualizing walker dynamics, branching, and energy evolution.`
