FROM node:22-bookworm-slim AS build

WORKDIR /src
COPY package.json package-lock.json ./
RUN npm ci

COPY tsconfig.json ./
COPY src ./src
RUN npm run build

FROM node:22-bookworm-slim AS runtime

ENV NODE_ENV=production \
    HOST=0.0.0.0 \
    PORT=4111
WORKDIR /app

# `mastra build` emits a self-contained server bundle, including its runtime
# node_modules. Keeping source and development dependencies out of this image
# also prevents the planning service from becoming an execution workspace.
COPY --from=build --chown=node:node /src/.mastra/output/ ./
RUN mkdir -p /app/data && chown node:node /app/data

USER node
EXPOSE 4111
CMD ["node", "index.mjs"]
