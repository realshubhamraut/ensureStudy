# Docker Volume Locations

## Where Data is Stored

When running with `docker-compose`, each database stores its data in Docker volumes.

### Volume Mappings

| Service | Volume Name | Container Path | Purpose |
|---------|-------------|----------------|---------|
| PostgreSQL | `postgres_data` | `/var/lib/postgresql/data` | SQL database files |
| Redis | `redis_data` | `/data` | RDB/AOF persistence |
| Qdrant | `qdrant_storage` | `/qdrant/storage` | Vector indices & snapshots |
| MongoDB | `mongo_data` | `/data/db` | BSON document storage |
| Cassandra | `cassandra_data` | `/var/lib/cassandra` | SSTable files |
| Kafka | `kafka_data` | `/var/lib/kafka/data` | Log segments |
| Zookeeper | `zookeeper_data` | `/var/lib/zookeeper/data` | ZK data |
| Zookeeper | `zookeeper_log` | `/var/lib/zookeeper/log` | Transaction logs |
| MLflow | `mlflow_artifacts` | `/mlflow/artifacts` | Model artifacts |

---

## Finding Docker Volumes on Host

### macOS (Docker Desktop)
Docker stores volumes in the Linux VM:
```bash
# List all volumes
docker volume ls

# Inspect a specific volume
docker volume inspect postgres_data

# Volumes are stored in the VM at:
# ~/Library/Containers/com.docker.docker/Data/vms/0/data/docker/volumes/
```

### Linux
```bash
# Volumes are stored at:
/var/lib/docker/volumes/<volume_name>/_data/

# Example:
ls /var/lib/docker/volumes/ensurestudy_postgres_data/_data/
```

---

## Volume Management Commands

### List All Project Volumes
```bash
docker volume ls | grep ensure
```

### Backup a Volume
```bash
# PostgreSQL
docker run --rm -v ensurestudy_postgres_data:/data -v $(pwd):/backup alpine \
  tar czf /backup/postgres_backup.tar.gz -C /data .

# Qdrant
docker run --rm -v ensurestudy_qdrant_storage:/data -v $(pwd):/backup alpine \
  tar czf /backup/qdrant_backup.tar.gz -C /data .
```

### Restore a Volume
```bash
# PostgreSQL
docker run --rm -v ensurestudy_postgres_data:/data -v $(pwd):/backup alpine \
  tar xzf /backup/postgres_backup.tar.gz -C /data
```

### Clean Up Volumes
```bash
# WARNING: This deletes all data!
docker-compose down -v

# Remove specific volume
docker volume rm ensurestudy_postgres_data
```

---

## Development vs Production

### Development (Current)
- Uses Docker volumes (ephemeral on container removal with `-v`)
- Data persists across container restarts
- Easy to reset with `docker-compose down -v`

### Production (Recommended)
- Use bind mounts to host directories
- Or use managed database services (RDS, Atlas, etc.)

Example bind mount for PostgreSQL:
```yaml
volumes:
  - ./datadir/volumes/postgres:/var/lib/postgresql/data
```

---

## Current Container Status

Check running containers:
```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

Expected output:
```
NAMES                    STATUS         PORTS
ensure-study-postgres    Up X hours     0.0.0.0:5432->5432/tcp
ensure-study-redis       Up X hours     0.0.0.0:6379->6379/tcp
ensure-study-qdrant      Up X hours     0.0.0.0:6333-6334->6333-6334/tcp
ensure-study-mongodb     Up X hours     0.0.0.0:27017->27017/tcp
ensure-study-cassandra   Up X hours     0.0.0.0:9042->9042/tcp
ensure-study-minio       Up X hours     0.0.0.0:9000->9000/tcp
```
