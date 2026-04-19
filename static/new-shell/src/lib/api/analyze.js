import { ApiError, getReadOnlyAuthToken } from './http.js';
import { logApiFailure } from '../observability.js';

const ANALYZE_UPLOAD_LIMIT_BYTES = 10 * 1024 * 1024;
const AUDIO_TYPE_TO_EXTENSION = [
  ['audio/webm', 'webm'],
  ['audio/mp4', 'mp4'],
  ['video/mp4', 'mp4'],
  ['audio/ogg', 'ogg'],
  ['audio/mpeg', 'mp3'],
  ['audio/mp3', 'mp3'],
  ['audio/wav', 'wav'],
  ['audio/wave', 'wav'],
  ['audio/x-wav', 'wav'],
  ['audio/x-m4a', 'm4a'],
  ['audio/m4a', 'm4a'],
];

function getErrorMessage(data, fallback) {
  if (typeof data?.detail === 'string' && data.detail.trim()) {
    return data.detail.trim();
  }

  if (Array.isArray(data?.detail) && data.detail.length > 0) {
    return data.detail
      .map((entry) => entry?.msg)
      .filter(Boolean)
      .join(' ') || fallback;
  }

  return fallback;
}

export function getAnalyzeAudioExtension(audioBlob) {
  const mimeType = String(audioBlob?.type || '').toLowerCase();

  if (!mimeType) {
    return 'webm';
  }

  const match = AUDIO_TYPE_TO_EXTENSION.find(([type]) => mimeType.includes(type));

  if (match) {
    return match[1];
  }

  throw new ApiError('This recorded take format is not supported for analysis.', {
    status: 400,
  });
}

function validateAnalyzeRequest({ sceneId, audioBlob }) {
  if (!sceneId) {
    throw new ApiError('No scene is selected for analyze submit.', { status: 400 });
  }

  if (!audioBlob) {
    throw new ApiError('Record a take before analyzing.', { status: 400 });
  }

  if (!audioBlob.size) {
    throw new ApiError('The recorded take is empty. Reset and record again before analyzing.', {
      status: 400,
    });
  }

  if (audioBlob.size > ANALYZE_UPLOAD_LIMIT_BYTES) {
    throw new ApiError('This take is larger than the analyze upload limit.', {
      status: 413,
    });
  }
}

export async function submitLegacyAnalyze({ sceneId, audioBlob, signal } = {}) {
  validateAnalyzeRequest({ sceneId, audioBlob });

  const token = getReadOnlyAuthToken();

  if (!token) {
    const error = new ApiError('Sign in before analyzing a take.', {
      status: 401,
      authRequired: true,
    });

    logApiFailure(error, {
      surface: 'analyze',
      path: '/api/submit',
      method: 'POST',
      sceneId,
      reason: 'missing-token',
    });
    throw error;
  }

  const extension = getAnalyzeAudioExtension(audioBlob);
  const formData = new FormData();
  formData.append('scene_id', sceneId);
  formData.append('audio', audioBlob, `recording.${extension}`);

  let response;

  try {
    response = await fetch('/api/submit', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
      },
      body: formData,
      signal,
    });
  } catch (error) {
    if (error?.name === 'AbortError') {
      throw error;
    }

    throw new ApiError('The analyze request could not reach the scoring service.', {
      status: 0,
    });
  }

  let data = null;

  try {
    data = await response.json();
  } catch {
    data = null;
  }

  if (!response.ok) {
    const error = new ApiError(getErrorMessage(data, 'Analyze failed.'), {
      status: response.status,
      authRequired: response.status === 401,
      rateLimited: response.status === 429,
    });

    logApiFailure(error, {
      surface: 'analyze',
      path: '/api/submit',
      method: 'POST',
      sceneId,
    });
    throw error;
  }

  return data;
}
