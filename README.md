# TaxiPulse

TaxiPulse is a taxi data analytics and machine learning project built with Python, Apache Spark and MongoDB. It cleans and aggregates taxi trip data with Spark, trains a prediction model, loads results into MongoDB, and presents them through a dashboard and control center. It also includes benchmarks that measure how the Spark pipeline scales.

## Project Structure

```
taxipulse/
├── app/
│   ├── control_center.py   # Control center for managing the project
│   ├── dashboard.py        # Dashboard for viewing results
│   ├── jobs.py             # Job handling for pipeline / ML tasks
│   └── mongo_tools.py      # MongoDB helper functions for the app
├── benchmark/
│   ├── run_benchmark.py    # Runs performance benchmarks
│   └── memory_scaling.py   # Tests how memory usage scales with data size
├── ml/
│   ├── train.py            # Trains the model
│   └── predict.py          # Makes predictions with the trained model
├── pipeline/
│   ├── clean.py            # Cleans the raw taxi data
│   ├── aggregate.py        # Aggregates the cleaned data
│   └── run_pipeline.py     # Runs the full data pipeline
├── serving/
│   └── load_mongo.py       # Loads results into MongoDB for serving
├── config.py               # Central configuration (paths and settings)
├── logging_utils.py        # Shared logging helpers
├── spark_session.py        # Creates and configures the Spark session
├── LICENSE                 # MIT license
└── README.md
```

## How the Pieces Fit Together

1. **pipeline** cleans the raw data and aggregates it using Spark.
2. **ml** trains a model on the processed data and uses it to predict.
3. **serving** loads the results into MongoDB.
4. **app** reads from MongoDB and shows the results in a dashboard, with a control center to manage jobs.
5. **benchmark** measures the speed and memory scaling of the pipeline.

All parts share `config.py` for settings, `spark_session.py` for the Spark session and `logging_utils.py` for logging.

## Tech Stack

- Python 3.8+
- Apache Spark (PySpark)
- MongoDB
- Git and GitHub for collaboration

## Prerequisites

- Python 3.8 or newer
- Java 8, 11 or 17 (required by Spark)
- A running MongoDB instance (local or hosted)
- Git

## Getting Started

1. Clone the repository
   ```
   git clone https://github.com/winnermarvi/taxipulse.git
   cd taxipulse
   ```
2. Create and activate a virtual environment
   ```
   python -m venv venv
   venv\Scripts\activate          # Windows
   source venv/bin/activate       # macOS / Linux
   ```
3. Install the dependencies
   ```
   pip install pyspark pymongo
   ```
   If a script fails with `ModuleNotFoundError`, install the library it names (for example `pip install pandas`).
4. Open `config.py` and update the data paths and MongoDB connection settings for your machine.

## Running the Project

Run these from inside the `taxipulse` folder, in this order:

```
python -m pipeline.run_pipeline      # 1. clean and aggregate the data
python -m ml.train                   # 2. train the model
python -m ml.predict                 # 3. generate predictions
python -m serving.load_mongo         # 4. load results into MongoDB
python -m app.dashboard              # 5. start the dashboard
python -m benchmark.run_benchmark    # optional: run benchmarks
python -m benchmark.memory_scaling   # optional: memory scaling test
```

Notes:
- If the dashboard is built with Streamlit, start it with `streamlit run app/dashboard.py` instead.
- If you get `ModuleNotFoundError: No module named 'taxipulse'`, run the commands from the parent folder with a `taxipulse.` prefix, for example `python -m taxipulse.pipeline.run_pipeline`.

## Contributing

1. Fork the repository
2. Create a branch: `git checkout -b my-feature`
3. Make your changes and commit: `git commit -m "Describe your change"`
4. Push the branch: `git push origin my-feature`
5. Open a Pull Request against `main` and describe what you changed

Keep each pull request focused on one change, and put code in the folder where it belongs.

## Team

| Name | Role | GitHub |
|------|------|--------|
| M Sufiyan Mithawala | Project owner | [M-Sufiyan-Mithawala](https://github.com/M-Sufiyan-Mithawala) |
| winnermarvi | Project owner | [winnermarvi](https://github.com/winnermarvi) |

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
