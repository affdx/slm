/**
 * API Route: POST /api/predict
 *
 * Proxies video upload to the FastAPI backend for sign language prediction.
 * This route acts as a BFF (Backend-for-Frontend) to:
 * - Hide the backend URL from the client
 * - Handle authentication if needed
 * - Add request logging/monitoring
 */

import { NextRequest, NextResponse } from 'next/server';

// Force dynamic rendering to ensure runtime env vars are read
export const dynamic = 'force-dynamic';

// Get backend URL at runtime (not build time)
function getBackendUrl(): string {
  return process.env.BACKEND_URL || 'http://localhost:8000';
}

export async function POST(request: NextRequest) {
  try {
    // Get the form data from the request
    const formData = await request.formData();
    const video = formData.get('video');
    const topK = formData.get('top_k') || '5';

    if (!video || !(video instanceof Blob)) {
      return NextResponse.json(
        { detail: 'No video file provided' },
        { status: 400 }
      );
    }

    // Create new FormData for backend request
    const backendFormData = new FormData();
    backendFormData.append('video', video);

    // Forward request to FastAPI backend
    const backendUrl = getBackendUrl();
    const backendResponse = await fetch(
      `${backendUrl}/predict?top_k=${topK}`,
      {
        method: 'POST',
        body: backendFormData,
      }
    );

    // Get response data
    const data = await backendResponse.json();

    // Return response with same status code
    return NextResponse.json(data, { status: backendResponse.status });
  } catch (error) {
    console.error('Prediction proxy error:', error);

    // Check if backend is unreachable
    if (error instanceof TypeError && error.message.includes('fetch')) {
      return NextResponse.json(
        { detail: 'Backend service unavailable' },
        { status: 503 }
      );
    }

    return NextResponse.json(
      { detail: 'Internal server error' },
      { status: 500 }
    );
  }
}

// Handle OPTIONS for CORS preflight
export async function OPTIONS() {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  });
}
