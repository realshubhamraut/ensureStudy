## Deployment

This document covers deployment configurations for development, staging, and production environments using Docker, Kubernetes, and cloud services.

### Deployment Architecture

```mermaid
graph TB
    subgraph "Load Balancer"
        LB[Nginx / Cloud LB]
    end
    
    subgraph "Application Tier"
        FE[Frontend Pods]
        Core[Core Service Pods]
        AI[AI Service Pods]
    end
    
    subgraph "Worker Tier"
        Celery[Celery Workers]
        SparkStream[Spark Streaming]
    end
    
    subgraph "Data Tier"
        PG[(PostgreSQL)]
        Qdrant[(Qdrant)]
        Redis[(Redis)]
        Kafka[(Kafka)]
    end
    
    LB --> FE
    LB --> Core
    LB --> AI
    
    Core --> PG
    Core --> Redis
    AI --> Qdrant
    AI --> Redis
    
    Celery --> PG
    Celery --> Redis
    SparkStream --> Kafka
```

### Docker Compose (Development)

```yaml
version: '3.8'

services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.dev
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app
      - /app/node_modules
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:5000
      - NEXT_PUBLIC_AI_SERVICE_URL=http://localhost:8000

  core-service:
    build:
      context: ./backend/core-service
      dockerfile: Dockerfile
    ports:
      - "5000:5000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/ensurestudy
      - REDIS_URL=redis://redis:6379
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
    depends_on:
      - postgres
      - redis

  ai-service:
    build:
      context: ./backend/ai-service
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - QDRANT_URL=http://qdrant:6333
      - REDIS_URL=redis://redis:6379
      - HUGGINGFACE_API_KEY=${HUGGINGFACE_API_KEY}
    depends_on:
      - qdrant
      - redis

  celery-worker:
    build:
      context: ./backend/core-service
    command: celery -A app.celery worker --loglevel=info
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/ensurestudy
      - CELERY_BROKER_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=ensurestudy
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  kafka:
    image: confluentinc/cp-kafka:latest
    ports:
      - "9092:9092"
    environment:
      - KAFKA_BROKER_ID=1
      - KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181
      - KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://kafka:29092,PLAINTEXT_HOST://localhost:9092
      - KAFKA_LISTENER_SECURITY_PROTOCOL_MAP=PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
    depends_on:
      - zookeeper

  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      - ZOOKEEPER_CLIENT_PORT=2181

volumes:
  postgres_data:
  qdrant_data:
  redis_data:
```

### Dockerfile Examples

**Frontend (Multi-stage)**

```dockerfile
# Build stage
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Production stage
FROM node:20-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs
EXPOSE 3000
ENV PORT 3000

CMD ["node", "server.js"]
```

**Core Service (Python)**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:create_app('production')"]
```

**AI Service (Python + CUDA optional)**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Kubernetes Manifests

**Namespace**

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ensurestudy
```

**Core Service Deployment**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: core-service
  namespace: ensurestudy
spec:
  replicas: 3
  selector:
    matchLabels:
      app: core-service
  template:
    metadata:
      labels:
        app: core-service
    spec:
      containers:
        - name: core-service
          image: ensurestudy/core-service:latest
          ports:
            - containerPort: 5000
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: db-secrets
                  key: url
            - name: REDIS_URL
              valueFrom:
                configMapKeyRef:
                  name: app-config
                  key: redis_url
          resources:
            requests:
              memory: "256Mi"
              cpu: "250m"
            limits:
              memory: "512Mi"
              cpu: "500m"
          livenessProbe:
            httpGet:
              path: /health
              port: 5000
            initialDelaySeconds: 10
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: 5000
            initialDelaySeconds: 5
            periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: core-service
  namespace: ensurestudy
spec:
  selector:
    app: core-service
  ports:
    - port: 5000
      targetPort: 5000
  type: ClusterIP
```

**AI Service Deployment**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-service
  namespace: ensurestudy
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ai-service
  template:
    metadata:
      labels:
        app: ai-service
    spec:
      containers:
        - name: ai-service
          image: ensurestudy/ai-service:latest
          ports:
            - containerPort: 8000
          env:
            - name: QDRANT_URL
              value: "http://qdrant:6333"
            - name: HUGGINGFACE_API_KEY
              valueFrom:
                secretKeyRef:
                  name: ai-secrets
                  key: huggingface_key
          resources:
            requests:
              memory: "1Gi"
              cpu: "500m"
            limits:
              memory: "2Gi"
              cpu: "1000m"
```

**Horizontal Pod Autoscaler**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: core-service-hpa
  namespace: ensurestudy
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: core-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

**Ingress**

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ensurestudy-ingress
  namespace: ensurestudy
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
    - hosts:
        - app.ensurestudy.com
        - api.ensurestudy.com
      secretName: ensurestudy-tls
  rules:
    - host: app.ensurestudy.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: frontend
                port:
                  number: 3000
    - host: api.ensurestudy.com
      http:
        paths:
          - path: /api/v1
            pathType: Prefix
            backend:
              service:
                name: core-service
                port:
                  number: 5000
          - path: /api/ai
            pathType: Prefix
            backend:
              service:
                name: ai-service
                port:
                  number: 8000
```

### Environment Configuration

**ConfigMap**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: ensurestudy
data:
  redis_url: "redis://redis:6379"
  qdrant_url: "http://qdrant:6333"
  kafka_brokers: "kafka:9092"
  log_level: "INFO"
```

**Secrets**

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: db-secrets
  namespace: ensurestudy
type: Opaque
stringData:
  url: "postgresql://user:password@postgres:5432/ensurestudy"

---
apiVersion: v1
kind: Secret
metadata:
  name: ai-secrets
  namespace: ensurestudy
type: Opaque
stringData:
  huggingface_key: "hf_..."
```

### CI/CD Pipeline (GitHub Actions)

```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Login to Docker Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push Core Service
        uses: docker/build-push-action@v5
        with:
          context: ./backend/core-service
          push: true
          tags: ghcr.io/${{ github.repository }}/core-service:${{ github.sha }}

      - name: Build and push AI Service
        uses: docker/build-push-action@v5
        with:
          context: ./backend/ai-service
          push: true
          tags: ghcr.io/${{ github.repository }}/ai-service:${{ github.sha }}

      - name: Build and push Frontend
        uses: docker/build-push-action@v5
        with:
          context: ./frontend
          push: true
          tags: ghcr.io/${{ github.repository }}/frontend:${{ github.sha }}

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Configure kubectl
        uses: azure/k8s-set-context@v3
        with:
          kubeconfig: ${{ secrets.KUBE_CONFIG }}

      - name: Update deployments
        run: |
          kubectl set image deployment/core-service \
            core-service=ghcr.io/${{ github.repository }}/core-service:${{ github.sha }} \
            -n ensurestudy
          kubectl set image deployment/ai-service \
            ai-service=ghcr.io/${{ github.repository }}/ai-service:${{ github.sha }} \
            -n ensurestudy
          kubectl set image deployment/frontend \
            frontend=ghcr.io/${{ github.repository }}/frontend:${{ github.sha }} \
            -n ensurestudy

      - name: Wait for rollout
        run: |
          kubectl rollout status deployment/core-service -n ensurestudy
          kubectl rollout status deployment/ai-service -n ensurestudy
          kubectl rollout status deployment/frontend -n ensurestudy
```

### Health Checks

```python
# Core Service health endpoints
@app.route('/health')
def health():
    return {'status': 'healthy'}, 200

@app.route('/ready')
def ready():
    try:
        # Check database connection
        db.session.execute(text('SELECT 1'))
        # Check Redis connection
        redis_client.ping()
        return {'status': 'ready'}, 200
    except Exception as e:
        return {'status': 'not ready', 'error': str(e)}, 503
```

### Resource Recommendations

| Service | CPU Request | CPU Limit | Memory Request | Memory Limit |
|---------|-------------|-----------|----------------|--------------|
| Frontend | 100m | 500m | 128Mi | 512Mi |
| Core Service | 250m | 500m | 256Mi | 512Mi |
| AI Service | 500m | 1000m | 1Gi | 2Gi |
| Celery Worker | 250m | 500m | 512Mi | 1Gi |
| PostgreSQL | 500m | 2000m | 1Gi | 4Gi |
| Qdrant | 500m | 1000m | 2Gi | 4Gi |
| Redis | 100m | 500m | 256Mi | 1Gi |
