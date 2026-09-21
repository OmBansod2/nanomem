# For registries and inspectors that build a server and introspect it over stdio
# (Glama, the MCP Inspector). Nothing here is needed to USE nanomem -- `pip
# install nanomem` and a one-line client config is the normal path.
#
# Built from the repository rather than from PyPI on purpose, so what is
# inspected is what is in this commit.
FROM python:3.12-slim

# numpy is the only dependency and ships manylinux wheels, so no compiler and no
# build-essential layer is needed.
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir . && rm -rf /root/.cache

# The vault is an ordinary file. A container has no home worth writing to, so it
# is named explicitly instead of defaulting to ~/.nanomem/memory.dat.
ENV NANOMEM_VAULT=/data/memory.dat
RUN mkdir -p /data
VOLUME ["/data"]

# Exec form, so SIGTERM reaches the server rather than a shell. That matters
# here: the server flushes on SIGTERM, and through 0.7.17 a client stopping it
# that way lost every write of the session.
ENTRYPOINT ["nanomem-mcp"]
