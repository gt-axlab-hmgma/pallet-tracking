/** Largest video the UI will send. Keep in step with proxyClientMaxBodySize in next.config.ts. */
export const MAX_VIDEO_BYTES = 2 * 1024 ** 3;

export const formatSize = (bytes: number) =>
  bytes >= 1024 ** 3
    ? `${(bytes / 1024 ** 3).toFixed(1)} GB`
    : `${Math.max(1, Math.round(bytes / 1024 ** 2))} MB`;
