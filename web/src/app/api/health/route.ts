/**
 * API Route: GET /api/health
 *
 * Health check endpoint for deployment health checks.
 * Since all inference is client-side, this just returns frontend status.
 */

import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

interface HealthStatus {
  status: 'healthy';
  timestamp: string;
  frontend: {
    status: 'ok';
    version: string;
  };
  inference: 'client-side';
}

export async function GET() {
  const health: HealthStatus = {
    status: 'healthy',
    timestamp: new Date().toISOString(),
    frontend: {
      status: 'ok',
      version: process.env.npm_package_version || '1.0.0',
    },
    inference: 'client-side',
  };

  return NextResponse.json(health, { status: 200 });
}
