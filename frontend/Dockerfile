FROM node:24-slim

WORKDIR /app
COPY frontend/package.json frontend/pnpm-lock.yaml* ./
RUN corepack enable && pnpm install
COPY frontend/ .
EXPOSE 5173
CMD ["pnpm", "dev", "--host", "0.0.0.0"]
