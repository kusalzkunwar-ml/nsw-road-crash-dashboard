# NSW Crash Risk Console: GIS + Machine Learning

An interactive decision-support dashboard for road-crash severity in New South Wales, built on
**92,082 Transport for NSW crash records (2020–2024)**. It has risk maps by local government area (LGA),
a fatal-risk model you can inspect and retrain in the browser, and a scenario scorer for "what if" road conditions.

🏆 **Team Winner, MIT IMPACT (Industry MIT Project and Communication Tracks) event, 1 October 2026.**
This was an industry capstone with **BAOAN Australia** for the Bachelor of Data Analytics at Melbourne
Institute of Technology, Sydney. I led the machine learning workstream.

### ▶ [Open the live dashboard](https://kusalzkunwar-ml.github.io/nsw-road-crash-dashboard/)

## What it shows

- **Crash patterns:** severity mix, fatality rate by speed zone, trends by year, month, day and hour, weather,
  vehicle type, crash configuration and first impact.
- **Map:** 9,400 sampled crashes plotted with Leaflet, plus the highest-risk LGAs ranked by model-predicted fatal risk.
- **The model:** odds multipliers per feature, ROC curve, a threshold slider showing the recall/precision trade-off,
  a model comparison, and a depth-3 decision tree written out as rules.
- **Retrain it live:** an Adam optimiser fits the logistic regression in your browser tab over the sampled crashes.
- **Scenario scoring:** pick a speed zone, weather, vehicle, lighting and location, and see the predicted fatal risk and what drives it.

## Key findings

- Fatal risk climbs with speed: 0.8% of crashes are fatal in zones up to 50 km/h, rising to **4.0% at 100 km/h and over**.
- **Unlit roads at high speed are the deadliest combination.** At 100 km/h and over, 6.9% of crashes with no street lighting are fatal, against 3.4% where lighting is on.
- Motorcycles, bicycles/scooters and heavy vehicles raise the odds of a fatal outcome the most.
- The highest predicted risk is in rural LGAs with high average speed zones, such as Upper Lachlan, Warrumbungle and Greater Hume.

## Models

| Model (dashboard features, random 75/25 split) | ROC AUC | Recall (fatal) |
|---|---:|---:|
| Random forest (300 trees) | 0.745 | 0.64 |
| Logistic regression (shipped in the page) | 0.732 | 0.57 |
| Decision tree (depth 3) | 0.692 | 0.64 |

The project's main **Random Forest**, trained on the full set of crash attributes with a time-based split
(train 2020–2022, test on unseen 2024), reaches **ROC AUC 0.826**. The dashboard ships the logistic
regression instead, because every coefficient can be read as an odds multiplier and it scores entirely in the
browser with no server.

Fatal crashes are rare (about 2.4% of the weighted sample), so the sample keeps every fatal crash and
re-weights the rest to restore the real class balance.

## Tech

HTML, CSS and vanilla JavaScript · Chart.js · Leaflet.js · data prepared in Python (pandas, scikit-learn) and
exported to `data.js` · hosted on GitHub Pages.

```
index.html   page layout and the model write-up
app.js       charts, map, model scoring, live retraining
data.js      aggregated data, sampled points and model coefficients (generated, do not edit)
styles.css   styling
```

Data: Transport for NSW, Crash and casualty statistics (2020–2024).

Built by Kushal Kunwar with the MIT capstone team · [LinkedIn](https://www.linkedin.com/in/kushal-kunwar-data/)
