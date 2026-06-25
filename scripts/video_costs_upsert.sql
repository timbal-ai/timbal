-- Upsert Costs rows for direct VideoGenerate models.
-- Safe to re-run: ON CONFLICT (name, unit) updates rate/description without duplicate keys.
-- Usage keys emitted by the tool: {name}:{unit} (see python/timbal/tools/video.py).

BEGIN;

INSERT INTO public."Costs" (name, description, unit, rate, currency)
VALUES
  -- google/veo-3.1
  ('google/veo-3.1', 'Per generated video second at 720p (audio included)', 'video_seconds_720p', 0.40, 'USD'),
  ('google/veo-3.1', 'Per generated video second at 1080p (audio included)', 'video_seconds_1080p', 0.40, 'USD'),
  ('google/veo-3.1', 'Count of videos generated (informational)', 'generations', 0, 'USD'),
  ('google/veo-3.1', 'Successful API requests (informational)', 'requests', 0, 'USD'),

  -- google/veo-3.1-fast
  ('google/veo-3.1-fast', 'Per generated video second at 720p (audio included)', 'video_seconds_720p', 0.10, 'USD'),
  ('google/veo-3.1-fast', 'Per generated video second at 1080p (audio included)', 'video_seconds_1080p', 0.12, 'USD'),
  ('google/veo-3.1-fast', 'Count of videos generated (informational)', 'generations', 0, 'USD'),
  ('google/veo-3.1-fast', 'Successful API requests (informational)', 'requests', 0, 'USD'),

  -- bytedance/seedance-2.0
  ('bytedance/seedance-2.0', 'Approximate per-second cost at 720p (BytePlus token billing proxy)', 'video_seconds_720p', 0.14, 'USD'),
  ('bytedance/seedance-2.0', 'Approximate per-second cost at 1080p (BytePlus token billing proxy)', 'video_seconds_1080p', 0.30, 'USD'),
  ('bytedance/seedance-2.0', 'BytePlus total_tokens when returned by the API (USD per token)', 'tokens', 0.000002, 'USD'),
  ('bytedance/seedance-2.0', 'Count of videos generated (informational)', 'generations', 0, 'USD'),
  ('bytedance/seedance-2.0', 'Successful API requests (informational)', 'requests', 0, 'USD'),

  -- bytedance/seedance-2.0-fast
  ('bytedance/seedance-2.0-fast', 'Approximate per-second cost at 720p (BytePlus token billing proxy)', 'video_seconds_720p', 0.10, 'USD'),
  ('bytedance/seedance-2.0-fast', 'Approximate per-second cost at 1080p (BytePlus token billing proxy)', 'video_seconds_1080p', 0.18, 'USD'),
  ('bytedance/seedance-2.0-fast', 'BytePlus total_tokens when returned by the API (USD per token)', 'tokens', 0.000002, 'USD'),
  ('bytedance/seedance-2.0-fast', 'Count of videos generated (informational)', 'generations', 0, 'USD'),
  ('bytedance/seedance-2.0-fast', 'Successful API requests (informational)', 'requests', 0, 'USD')
ON CONFLICT (name, unit) DO UPDATE SET
  description = EXCLUDED.description,
  rate = EXCLUDED.rate,
  currency = EXCLUDED.currency;

COMMIT;
