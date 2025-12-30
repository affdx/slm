/**
 * API Route: GET /api/glosses
 *
 * Proxies glosses list request to the FastAPI backend.
 * Returns the list of all supported Malaysian Sign Language glosses.
 */

import { NextResponse } from 'next/server';

// Force dynamic rendering to ensure runtime env vars are read
export const dynamic = 'force-dynamic';

// Get backend URL at runtime (not build time)
function getBackendUrl(): string {
  return process.env.BACKEND_URL || 'http://localhost:8000';
}

export async function GET() {
  try {
    const backendUrl = getBackendUrl();
    const backendResponse = await fetch(`${backendUrl}/glosses`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      // Cache for 1 hour since glosses don't change
      next: { revalidate: 3600 },
    });

    if (!backendResponse.ok) {
      return NextResponse.json(
        { detail: 'Failed to fetch glosses' },
        { status: backendResponse.status }
      );
    }

    const data = await backendResponse.json();

    // Return with cache headers
    return NextResponse.json(data, {
      status: 200,
      headers: {
        'Cache-Control': 'public, s-maxage=3600, stale-while-revalidate=86400',
      },
    });
  } catch (error) {
    console.error('Glosses proxy error:', error);

    return NextResponse.json(
      { detail: 'Backend service unavailable' },
      { status: 503 }
    );
  }
}
