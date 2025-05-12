# AkshuAI Documentation

Welcome to the documentation for AkshuAI, an open-source platform for emulating AGI.

This document provides an overview of the project, its architecture, and how to get started.

## Table of Contents

- [Getting Started](#getting-started)
- [Architecture Overview](architecture.md)
- [ Module Documentation](modules.md)
- [Contribution Guide](CONTRIBUTING.md)
- [API Documentation](api.md)

---

## Getting Started

This section will guide you through setting up the AkshuAI development environment.

### Prerequisites

Before you begin, ensure you have the following installed:

*   **Python 3.8+**: Required for the backend services.
*   **Docker and Docker Compose**: Used for containerizing and orchestrating the microservices.
*   **Redis**: Required for the Memory module and caching in the Orchestrator.

### Setup Steps

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/Abhaythakurrr/akshuai.git
    cd akshuai
    ```

2.  **Environment Variables:**

    Copy the example environment file and update it with your settings.

    ```bash
    cp .env.example .env
    ```

    Open the `.env` file and configure the service URLs, timeouts, and any module-specific settings (like the language model name) as needed for your environment. When running locally with Docker Compose, the default `localhost` URLs should generally work.

3.  **Install Dependencies:**

    Each service has its own `requirements.txt` file. Navigate to each service directory and install the dependencies.

    ```bash
    # Install dependencies for the Orchestrator service
    cd akshu-ai/services/orchestrator
    pip install -r requirements.txt

    # Install dependencies for the Language service
    cd ../language
    pip install -r requirements.txt

    # Install dependencies for the Persona service
    cd ../persona
    pip install -r requirements.txt

    # Repeat for other services (memory, vision, audio, execution) as they are implemented.
    # Remember to return to the project root after installing dependencies for each service.
    cd ../../..
    ```

4.  **Build and Run with Docker Compose:**

    Navigate back to the `akshu-ai` directory (where `docker-compose.yml` is located) and build and run the services.

    ```bash
    cd akshu-ai
    docker-compose build
    docker-compose up
    ```

    This will build the Docker images for the services with Dockerfiles and start the containers.

### Next Steps

*   Explore the [Architecture Overview](architecture.md).
*   Dive into [Module Documentation](modules.md) for details on individual services.
*   See the [API Documentation](api.md) to understand how to interact with the services.
*   Learn how to [Contribute](CONTRIBUTING.md) to the project.

---

This documentation is a work in progress. More detailed guides and explanations will be added over time.