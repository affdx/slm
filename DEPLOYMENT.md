# Deployment Guide - Sevalla

This guide explains how to deploy the Malaysian Sign Language (MSL) Translation System to Sevalla.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         SEVALLA                                  │
│                                                                  │
│  ┌──────────────────────┐      ┌─────────────────────────────┐  │
│  │   msl-web            │      │   msl-api                   │  │
│  │   (Next.js)          │ ───► │   (FastAPI + PyTorch)       │  │
│  │   Application        │      │   Application               │  │
│  │                      │      │                             │  │
│  │   Port: 3000         │      │   Port: 8000                │  │
│  └──────────────────────┘      └─────────────────────────────┘  │
│            │                              │                      │
│            └──────── Private Network ─────┘                      │
│                  (internal connection)                           │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. **Sevalla Account**: Sign up at [sevalla.com](https://sevalla.com) (get $50 free credit)
2. **Git Repository**: Push your code to GitHub, GitLab, or Bitbucket
3. **Model Files**: Ensure `models/best.pt` and `models/class_mapping.json` exist

## Deployment Steps

### Step 1: Deploy the FastAPI Backend (msl-api)

1. **Go to Sevalla Dashboard** → Applications → Add Application

2. **Configure Application**:
   | Setting | Value |
   |---------|-------|
   | Name | `msl-api` |
   | Git Repository | Select your repo |
   | Branch | `main` |
   | Root Directory | `/` (root) |
   | Build Type | **Dockerfile** |
   | Dockerfile Path | `Dockerfile` |

3. **Set Resources** (minimum recommended):
   | Resource | Value |
   |----------|-------|
   | CPU | 1 vCPU |
   | RAM | **2 GB** (PyTorch + MediaPipe need this) |
   | Instances | 1 |

4. **Environment Variables**:
   | Variable | Value |
   |----------|-------|
   | `MODEL_PATH` | `models/best.pt` |
   | `CLASS_MAPPING_PATH` | `models/class_mapping.json` |
   | `LOG_LEVEL` | `INFO` |

5. **Health Check** (optional but recommended):
   | Setting | Value |
   |---------|-------|
   | Path | `/health` |
   | Initial Delay | `60s` (model loading time) |
   | Interval | `30s` |

6. **Click Deploy** and wait for the build to complete (~5-10 minutes first time)

7. **Note the Internal URL**: After deployment, go to **Networking** and note the internal hostname (e.g., `msl-api-xxxxx.internal`)

---

### Step 2: Deploy the Next.js Frontend (msl-web)

1. **Go to Sevalla Dashboard** → Applications → Add Application

2. **Configure Application**:
   | Setting | Value |
   |---------|-------|
   | Name | `msl-web` |
   | Git Repository | Select your repo |
   | Branch | `main` |
   | Root Directory | `/web` |
   | Build Type | **Dockerfile** |
   | Dockerfile Path | `Dockerfile` |

3. **Set Resources**:
   | Resource | Value |
   |----------|-------|
   | CPU | 0.5 vCPU |
   | RAM | 512 MB |
   | Instances | 1 |

4. **Environment Variables**:
   | Variable | Value |
   |----------|-------|
   | `BACKEND_URL` | Internal URL from Step 1 (e.g., `http://msl-api-xxxxx.internal:8000`) |
   | `NEXT_PUBLIC_API_URL` | Your public frontend URL (optional) |

5. **Click Deploy**

---

### Step 3: Connect Applications via Private Network

1. Go to **msl-web** → **Networking** → **Internal Connections**

2. Click **Add Connection** and select **msl-api**

3. This creates a private network connection between the two applications

4. Update the `BACKEND_URL` environment variable in msl-web to use the internal hostname

---

### Step 4: Configure Domain (Optional)

1. Go to **msl-web** → **Domains**

2. Add your custom domain or use the provided Sevalla subdomain

3. SSL certificates are automatically provisioned

---

## Environment Variables Reference

### FastAPI Backend (msl-api)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MODEL_PATH` | No | `models/best.pt` | Path to trained model |
| `CLASS_MAPPING_PATH` | No | `models/class_mapping.json` | Path to class mapping |
| `LOG_LEVEL` | No | `INFO` | Logging level |

### Next.js Frontend (msl-web)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `BACKEND_URL` | **Yes** | `http://localhost:8000` | Internal URL to FastAPI |
| `NEXT_PUBLIC_API_URL` | No | - | Public API URL (if needed) |

---

## Cost Estimation

| Component | Tier | Estimated Cost |
|-----------|------|----------------|
| msl-api (FastAPI) | 1 vCPU, 2GB RAM | ~$15/month |
| msl-web (Next.js) | 0.5 vCPU, 512MB RAM | ~$5/month |
| **Total** | | **~$20/month** |

*Note: Sevalla offers $50 free credit for new accounts*

---

## Local Development

### Running Both Services Locally

```bash
# Terminal 1: Start FastAPI backend
source venv/bin/activate
uvicorn src.inference.api:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Start Next.js frontend
cd web
npm run dev
```

### Testing Docker Builds Locally

```bash
# Build and run FastAPI
docker build -t msl-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models msl-api

# Build and run Next.js
cd web
docker build -t msl-web .
docker run -p 3000:3000 -e BACKEND_URL=http://host.docker.internal:8000 msl-web
```

---

## Troubleshooting

### Common Issues

#### 1. Model not loading (503 error)
- **Cause**: Not enough memory for PyTorch + MediaPipe
- **Fix**: Increase RAM to 2GB minimum

#### 2. Build fails on Dockerfile
- **Cause**: Missing dependencies or wrong paths
- **Fix**: Check build logs, ensure `models/` directory is included

#### 3. Backend unreachable from frontend
- **Cause**: Wrong `BACKEND_URL` or no internal connection
- **Fix**: 
  1. Verify internal connection is set up
  2. Use correct internal hostname (check Networking tab)
  3. Ensure port 8000 is correct

#### 4. Slow first request (~10s)
- **Cause**: Model cold start (loading into memory)
- **Fix**: This is expected on first request. Enable health checks with longer initial delay.

#### 5. CORS errors
- **Cause**: Direct browser calls to backend
- **Fix**: Use the Next.js API proxy routes (`/api/predict`) instead of calling backend directly

### Viewing Logs

1. Go to your application in Sevalla Dashboard
2. Click **Runtime Logs**
3. Filter by severity (Error, Warning, Info)

---

## Performance Optimization

### For Production

1. **Enable CDN**: Go to msl-web → CDN → Enable
2. **Enable Edge Caching**: For static assets
3. **Scale horizontally**: Add more instances during peak traffic
4. **Use health checks**: For zero-downtime deployments

### Recommended Settings

```
msl-api:
  - Instances: 1-2
  - Auto-scaling: Enable if high traffic expected
  - Health check: /health with 60s initial delay

msl-web:
  - Instances: 1-2
  - CDN: Enabled
  - Edge caching: Enabled for static assets
  - Health check: /api/health with 30s initial delay
```

---

## Updating Your Application

### Automatic Deployments

1. Push to your main branch
2. Sevalla automatically builds and deploys
3. Zero-downtime deployment with health checks

### Manual Deployments

1. Go to your application
2. Click **Deployments** → **Deploy Now**
3. Select branch/commit to deploy

---

## Support

- **Sevalla Docs**: [docs.sevalla.com](https://docs.sevalla.com)
- **Sevalla Discord**: [discord.gg/sevalla](https://discord.gg/sevalla)
- **Project Issues**: Create an issue in this repository
