# CMPT 310 Location Interface

Local website prototype for the CMPT 310 restaurant location success project.

The interface lets a user choose a Metro Vancouver restaurant location, select which project models to compare, adjust the input features, and read the two main prediction outputs:

- Expected Yelp rating
- Successful or not successful classification

The generated feature table shows the input values used by the selected models. It is not a separate model output.

## Technology Stack

- TypeScript and TSX for the React interface
- CSS/Tailwind tooling for styling
- JavaScript/Node.js for scripts and build tooling
- Vinext/Vite for the frontend build
- Leaflet with OpenStreetMap tiles for the map

## Run Locally

```bash
npm install
npm run dev
```

Open `http://localhost:3000/`.

## Project Context

Reference repository: `https://github.com/preethi-ca/CMPT310_Project`

The current version uses a browser-side TypeScript adapter generated from the project CSV files for the demo interface. The project repository contains the Python scripts for Ridge regression, Decision Tree regression, KNN classification, and XGBoost classification. When the team exports API-ready trained model artifacts, the interface can replace the local adapter with a server or API route that loads those artifacts directly.

To refresh the bundled model data after the project repo changes:

```bash
git clone https://github.com/preethi-ca/CMPT310_Project.git ../CMPT310_Project
node scripts/import-project-model-data.mjs ../CMPT310_Project
```

## Deployment Notes

The current app can be deployed as a static/reactive site because prediction logic runs in TypeScript in the browser for the class demo. Vercel is a good option for a public demo.
