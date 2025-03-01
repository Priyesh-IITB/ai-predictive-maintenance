# AI Predictive Maintenance System

A machine learning system for predicting valve failures and providing maintenance recommendations using real-time sensor data.

## Features

- Real-time valve failure prediction based on sensor data (vibration, temperature, pressure)
- Interactive dashboard for monitoring valve health and system metrics
- Root cause analysis for potential failures
- Alert system integration (Slack, SMS via Twilio)
- Prometheus and Grafana integration for monitoring
- Docker-based deployment for quick setup

## Architecture

The system consists of the following components:

- **API Service**: FastAPI application providing prediction endpoints
- **Dashboard**: Streamlit web application for visualization and interaction
- **RCA Engine**: Root cause analysis service
- **Prometheus**: Metrics collection and storage
- **Grafana**: Metrics visualization and alerting

## Quickstart

1. Clone the repository:
   ```
   git clone https://github.com/Priyesh-IITB/ai-predictive-maintenance.git
   cd ai-predictive-maintenance
   ```

2. Configure environment variables in `.env` file

3. Start the services using Docker Compose:
   ```
   docker-compose up -d
   ```

4. Access the services:
   - Dashboard: http://localhost:8501
   - API: http://localhost:8000
   - Grafana: http://localhost:3000 (admin/admin)
   - Prometheus: http://localhost:9090

## Online Deployment Options

### Render

1. Create a free account at [render.com](https://render.com)
2. Create a new Web Service and connect to your GitHub repository
3. Configure the environment variables
4. Deploy the application

### Railway

1. Create a free account at [railway.app](https://railway.app)
2. Create a new project and connect to your GitHub repository
3. Configure the environment variables
4. Deploy the application

### Fly.io

1. Create a free account at [fly.io](https://fly.io)
2. Install the Fly CLI: `curl -L https://fly.io/install.sh | sh`
3. Authenticate: `flyctl auth login`
4. Launch the app: `flyctl launch`
5. Deploy the app: `flyctl deploy`

## API Endpoints

- `GET /health`: System health check
- `POST /predict`: Predict valve failure risk
- `POST /rca`: Perform root cause analysis
- `GET /metrics`: Get system metrics
- `GET /prometheus`: Prometheus metrics endpoint

## Development

### Local Setup

1. Create a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the API:
   ```
   uvicorn scripts.api.app:app --reload
   ```

4. Run the dashboard:
   ```
   streamlit run scripts.dashboard.app
   ```

## License

MIT License

## Contact

Priyesh Verma - priyesh@example.com