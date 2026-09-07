# Minimal image for running an agent built on this SDK.
#
# Build:  docker build -t agent-engine .
# Run:    docker run --rm -e DEEPSEEK_API_KEY agent-engine
#
# The default entrypoint just verifies the package imports; a real product image would set
# its own entrypoint that constructs engine.Agent (or the Coordinator) and runs a task.

FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml ./
COPY engine ./engine
RUN pip install --no-cache-dir .

# Non-root user; /work is the agent's writable workspace (map Agent(workdir="/work")).
RUN useradd -m -u 10001 agent && mkdir -p /work && chown agent:agent /work
USER agent
WORKDIR /work

ENTRYPOINT ["python", "-c", "import engine; print('engine', engine.__version__, 'ready')"]
