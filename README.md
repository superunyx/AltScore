# AltScore

AltScore is a federated learning fintech project. This monorepo is structured to support a distributed machine learning architecture while keeping the backend, clients, and shared assets organized.

## Architecture

- **server/**: The aggregation server built with FastAPI and Flower. It coordinates the federated learning process, sending model updates to clients and aggregating their trained weights.
- **simulation/**: Python scripts that simulate mobile clients. This serves as a safe fallback path and a rapid testing environment for the federated learning loop without needing physical devices.
- **mobile/**: A React Native (Expo) application that acts as the real-world mobile client for participating in the federated learning process on user devices.
- **shared/**: Shared model architecture definitions (Keras/TensorFlow) that ensure the server and all clients (simulated and mobile) use identical model structures.
- **data/**: Scripts for generating synthetic financial data used for training, testing, and simulation.
