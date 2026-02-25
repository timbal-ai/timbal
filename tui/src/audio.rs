use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use color_eyre::Result;
use symphonia::core::audio::SampleBuffer as SymphoniaSampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Shared ring buffer of PCM samples for visualization.
pub type SampleBuffer = Arc<Mutex<Vec<f32>>>;

/// Decode an MP3 file on a background thread, streaming samples into a shared buffer
/// at real-time pace (no audio playback).
pub fn decode_mp3(path: &Path) -> Result<SampleBuffer> {
    let buffer: SampleBuffer = Arc::new(Mutex::new(Vec::with_capacity(8192)));
    let buf_clone = Arc::clone(&buffer);
    let path = PathBuf::from(path);

    thread::spawn(move || {
        if let Err(e) = decode_loop(&path, &buf_clone) {
            eprintln!("audio decode error: {e}");
        }
    });

    Ok(buffer)
}

fn decode_loop(path: &Path, buffer: &SampleBuffer) -> Result<()> {
    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    hint.with_extension("mp3");

    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;

    let mut format = probed.format;
    let track = format.default_track().ok_or_else(|| color_eyre::eyre::eyre!("no audio track"))?;
    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(44100) as f64;

    let mut decoder =
        symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;

    let mut samples_written: u64 = 0;
    let start = Instant::now();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(_) => break,
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(_) => continue,
        };

        let spec = *decoded.spec();
        let num_frames = decoded.frames();
        let channels = spec.channels.count();

        let mut sample_buf = SymphoniaSampleBuffer::<f32>::new(num_frames as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);
        let samples = sample_buf.samples();

        // Keep stereo interleaved (L, R, L, R, ...).
        // If mono, duplicate to fake stereo.
        let stereo: Vec<f32> = if channels >= 2 {
            // Take first two channels interleaved.
            (0..num_frames)
                .flat_map(|i| {
                    let l = samples[i * channels];
                    let r = samples[i * channels + 1];
                    [l, r]
                })
                .collect()
        } else {
            // Mono → duplicate.
            samples.iter().flat_map(|&s| [s, s]).collect()
        };

        if let Ok(mut buf) = buffer.lock() {
            buf.extend_from_slice(&stereo);
            // Keep last ~8192 samples (4096 stereo frames).
            if buf.len() > 16384 {
                let excess = buf.len() - 8192;
                buf.drain(..excess);
            }
        }

        samples_written += num_frames as u64;

        // Throttle to real-time so the visualization progresses naturally.
        let expected = Duration::from_secs_f64(samples_written as f64 / sample_rate);
        let actual = start.elapsed();
        if expected > actual {
            thread::sleep(expected - actual);
        }
    }

    Ok(())
}
