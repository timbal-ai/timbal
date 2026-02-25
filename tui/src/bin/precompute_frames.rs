//! One-shot tool: decode an MP3 and write precomputed vectorscope frames to a binary file.
//! Usage: cargo run --bin precompute-frames -- <input.mp3> <output.bin>
//!
//! Binary format:
//!   u32 LE: number of frames
//!   u32 LE: points per frame (FRAME_SIZE)
//!   For each frame:
//!     FRAME_SIZE × (f32 LE, f32 LE) — (left, right) pairs

use std::fs::File;
use std::io::Write;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: precompute-frames <input.mp3> <output.bin>");
        std::process::exit(1);
    }

    let input = &args[1];
    let output = &args[2];

    eprintln!("Decoding {input}...");
    let frames = timbal_tui::audio::precompute_frames(Path::new(input))
        .expect("Failed to decode MP3");

    eprintln!("Got {} frames", frames.len());

    let frame_size = if frames.is_empty() { 0u32 } else { frames[0].len() as u32 };

    let mut file = File::create(output).expect("Failed to create output file");

    // Header.
    file.write_all(&(frames.len() as u32).to_le_bytes()).unwrap();
    file.write_all(&frame_size.to_le_bytes()).unwrap();

    // Frames.
    for frame in &frames {
        for &(l, r) in frame {
            file.write_all(&(l as f32).to_le_bytes()).unwrap();
            file.write_all(&(r as f32).to_le_bytes()).unwrap();
        }
    }

    eprintln!("Wrote {output} ({} bytes)", 8 + frames.len() * frame_size as usize * 8);
}
