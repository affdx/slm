/**
 * API Route: GET /api/health
 *
 * Health check endpoint for Sevalla zero-downtime deployments.
 * Returns the health status of both the frontend and backend services.
 */

import { NextResponse } from 'next/server';

// Force dynamic rendering to ensure runtime env vars are read
export const dynamic = 'force-dynamic';

// Get backend URL at runtime (not build time)
function getBackendUrl(): string {
  return process.env.BACKEND_URL || 'http://localhost:8000';
}

interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  frontend: {
    status: 'ok';
    version: string;
  };
  backend?: {
    status: string;
    model_loaded: boolean;
    device: string;
    num_classes: number;
  };
  error?: string;
}

export async function GET() {
  const health: HealthStatus = {
    status: 'healthy',
    timestamp: new Date().toISOString(),
    frontend: {
      status: 'ok',
      version: process.env.npm_package_version || '1.0.0',
    },
  };

  try {
    // Check backend health
    const backendUrl = getBackendUrl();
    const backendResponse = await fetch(`${backendUrl}/health`, {
      method: 'GET',
      // Short timeout for health checks
      signal: AbortSignal.timeout(5000),
      cache: 'no-store',
    });

    if (backendResponse.ok) {
      health.backend = await backendResponse.json();
      // If backend is degraded (model not loaded), mark overall as degraded
      if (health.backend?.status === 'degraded') {
        health.status = 'degraded';
      }
    } else {
      health.status = 'degraded';
      health.error = 'Backend returned non-OK status';
    }
  } catch (error) {
    // Backend unreachable - service is degraded but frontend still works
    health.status = 'degraded';
    health.error = 'Backend service unreachable';
    console.warn('Health check: Backend unreachable', error);
  }

  // Return 200 for healthy/degraded (allows deployment to proceed)
  // Return 503 only for critical failures
  const statusCode = health.status === 'unhealthy' ? 503 : 200;

  return NextResponse.json(health, { status: statusCode });
}
