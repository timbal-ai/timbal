import openai
import os
import subprocess
import wave

# --- Configuration ---
# Load your OpenAI API key from an environment variable for better security.
# Set the OPENAI_API_KEY environment variable before running the script.
# Example on macOS/Linux: export OPENAI_API_KEY='your_key_here'
# Example on Windows: set OPENAI_API_KEY=your_key_here
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
openai.api_key = api_key

# --- Audio Recording Configuration ---
RECORD_SECONDS = 5       # Duration of recording
TEMP_AUDIO_FILE = "temp_recording.wav"

# --- Function to record audio using SoX ---
def record_audio(file_path):
    """Records audio from the microphone using SoX and saves it to a file."""
    # Note: This requires the 'sox' command-line utility.
    # On macOS: brew install sox
    # On Debian/Ubuntu: sudo apt-get install sox
    print("Recording for 5 seconds...")
    command = [
        'sox',
        '-d',  # Default audio device
        '-t', 'wav',
        file_path,
        'trim', '0', str(RECORD_SECONDS)
    ]
    
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print("Recording finished.")
    except FileNotFoundError:
        print("\nError: 'sox' command not found.")
        print("Please install SoX (Sound eXchange) on your system.")
        print("On macOS, run: brew install sox")
        exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\nError during recording with SoX: {e}")
        print(f"SoX stderr: {e.stderr}")
        exit(1)

# --- Function to transcribe audio ---
def transcribe_audio(file_path):
    """Transcribes an audio file using OpenAI's Whisper model."""
    if not os.path.exists(file_path):
        print(f"Error: The audio file '{file_path}' was not found.")
        return

    print(f"Attempting to transcribe the file: {file_path}")
    try:
        with open(file_path, "rb") as audio_file:
            response = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )

        transcribed_text = response.text
        print("\n--- Transcription Result ---")
        print(f"Transcribed Text: {transcribed_text}")

        if transcribed_text:
            print("\nOpenAI's STT model seems to be working correctly.")
        else:
            print("\nOpenAI's STT model worked, but no text was detected.")
            print("This might be due to a silent or unclear audio recording.")

    except openai.APIError as e:
        print(f"\nOpenAI API Error: {e}")
        print("Please check your API key or OpenAI account status.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        print("Please ensure the 'openai' library is installed correctly.")

# --- Run the script ---
if __name__ == "__main__":
    record_audio(TEMP_AUDIO_FILE)
    transcribe_audio(TEMP_AUDIO_FILE)
    
    # Clean up the temporary file
    if os.path.exists(TEMP_AUDIO_FILE):
        os.remove(TEMP_AUDIO_FILE)
        print(f"\nTemporary file '{TEMP_AUDIO_FILE}' has been deleted.")