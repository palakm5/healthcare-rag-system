# Healthcare RAG System Setup and Usage Guide

## Project Overview

This is a Retrieval-Augmented Generation (RAG) system designed for healthcare applications. The system ingests, processes, and retrieves information from various healthcare data sources to generate relevant responses.

## Prerequisites

Before setting up the project, ensure you have the following installed:

- Python 3.9+
- PostgreSQL (for data storage)
- Docker (optional, for containerized deployment)
- Git

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/palakm5/healtcare-rag-system.git
   cd healthcare-rag-system
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up the database:
   - Create a PostgreSQL database
   - Update the database connection details in `config/settings.py`
   - Run the schema setup:
     ```bash
     psql -U your_username -d your_database -f structured-data/full_schema.sql
     ```

## Configuration

1. Environment variables:
   - Create a `.env` file in the root directory
   - Add required environment variables (see `.env.example` for reference)

2. Customize settings in `config/settings.py` as needed

## Running the Application

1. Start the ingestion pipeline:
   ```bash
   python ingestion/ingest_pipeline.py
   ```

2. Run the retrieval service:
   ```bash
   python retrieval/search/retriever.py
   ```

3. Start the generation service:
   ```bash
   python generation/generator.py
   ```

4. (Optional) Run the CLI interface:
   ```bash
   python cli/rag_cli.py
   ```


## Additional Resources

- [Project Documentation](docs/iteration-1.md)
- [Data Sources](structured-data/)
- [Unstructured Data](unstructured-data/)

## Troubleshooting

- Check logs in the respective service directories
- Ensure all dependencies are properly installed
- Verify database connection settings
- Consult the project documentation for specific issues