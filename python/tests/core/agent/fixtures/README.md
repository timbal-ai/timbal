We've found problems with openAI's API when using _Opus_ encoded audio files (2025-05-12).

Check the audio file properties with:

```bash
ffprobe -hide_banner -loglevel error -show_streams -show_format <filename>.wav
```

Convert to an encoding that works with openai:

```bash
ffmpeg -i <input_filename>.wav -acodec pcm_s16le -ac 1 -ar 16000 <output_filename>.wav
```
