---
title: Context 
sidebar: 'docsSidebar_v2'
---
import CodeBlock from '@site/src/theme/CodeBlock';

# Context

<h2 className="subtitle" style={{marginTop: '-17px', fontSize: '1.1rem', fontWeight: 'normal'}}>
Store and access information across different steps. 
This enables data sharing, state management, and complex data flows between workflow steps.
</h2>

---



## Step Variables

Every step in a workflow has access to two built-in variables:

- **`.input`**: Contains all the parameters passed to the step. They are accessed through their name.
- **`.output`**: Contains the value(s) returned by the step. Can be a single value, dictionary, array, or custom class.

Additionally, **custom variables** can be linked to a step using the [`get_run_context().set_data`](#the-get_run_context-function) function.

<CodeBlock language="python" highlight="6" code ={`import asyncio
import datetime
from timbal import Workflow, get_run_context

async def process_user_data(user_id: str):
    get_run_context().set_data(".user_status", "active")
    return f"Processed user: {user_id}"

async def send_notification(message: str, user: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return [message, user, timestamp]


workflow = (Workflow(name='user_workflow')
    .step(process_user_data, user_id="user_123")
    .step(send_notification, message="Welcome!", user="user_123")
)`}/>

After the workflow runs, each step has the following variables:

### Step `process_user_data`
<CodeBlock language="python" code={`.input.user_id       = 'user_123'
.output              = 'Processed user: user_123'
.user_status         = 'active'`}/>

### Step `send_notification`
<CodeBlock language="python" code={`.input.message       = 'Welcome!'
.input.user          = 'user_123'
.output[0]           = 'Welcome!'
.output[1]           = 'user_123'
.output[2]           = '2025-01-15 14:30:25'`}/>

<!-- - **Custom variables** (like `.user_status`) are created using `get_run_context().set_data()` and persist across all steps
- Variables are accessed using dot notation: `.input.parameter_name`, `.output`, `.custom_variable` -->


<!-- ## The `get_run_context()` Function -->

## Accessing the Context
### The `get_run_context()` Function
This function is the primary way to interact with workflow context data. It provides access to the current step's context and allows you to read and write data that can be shared across workflow steps.

- **`get_data(path)`**:
  - Returns the value from the specified path.
  - Example: `get_run_context().get_data(".input.user_id")`

- **`set_data(path, value)`**:
  - Creates or updates the value at the given path.
  - Data persists and can be accessed by other steps.
  - Example: `get_run_context().set_data(".user_status", "active")`

<!-- ### Context Paths
Context paths use a hierarchical structure where:
- `.` refers to the current step
- `..` refers to the previous step  
- `step_name` allows to refer data from a neighbour step


<!-- Steps can access their own data as well as the data of other steps in the workflow. All steps within a workflow share data with each other, making it accessible across the entire workflow. -->

### Variable Access
Context paths use a hierarchical structure. Within a workflow, each step has access to its own data as well as the data produced by other steps, since information is shared across the entire process.
<!-- Variable access uses a path-based notation to reference data from different steps: -->

- **Current step**: Use `.` to access variables from the current step
  - `.input.parameter_name`
  - `.output`
  - `.custom_variable`

- **Parent step**: Use `..` to access data from the parent step in the workflow
  - `..output`
  - `..custom_variable`

- **Neighbour steps**: Use the step name to access data from any neighbour step
  - `step_name.output`
  - `step_name.custom_variable`



<!-- To access context variables within your step functions, use the `get_run_context()` function: -->

<CodeBlock language="python" code={`async def check_status():
    status = get_run_context().get_data("process_user_data.user_status")
    print(f"Your current status is: {status}")


workflow = (Workflow(name='user_workflow')
    .step(process_user_data, user_id="user_123")
    .step(send_notification, message="Welcome!", user="user_123")
    .step(check_status)
)`}/>

<!-- ## Practical Case: Input/Output Format Conversion

A common use case is transforming data formats between workflow steps. Here's an example showing how to convert audio input to text using `.input` and `.output`:

<CodeBlock language="python" code={`import asyncio
from timbal import Workflow, get_run_context

async def transcribe_audio(audio_file_path: str):
    """Step 1: Convert audio file to text using speech recognition"""
    # Simulate audio transcription (replace with actual speech recognition library)
    transcribed_text = f"Transcribed audio from {audio_file_path}: Hello, this is a test recording."
    
    # Store the original audio path for reference
    get_run_context().set_data(".audio_path", audio_file_path)
    
    return transcribed_text

async def process_text(text: str):
    """Step 2: Process the transcribed text"""
    # Access the transcribed text from previous step's output
    context = get_run_context()
    original_audio = context.get_data("transcribe_audio.audio_path")
    
    # Process the text (e.g., extract keywords, sentiment analysis, etc.)
    word_count = len(text.split())
    processed_data = {
        "original_audio": original_audio,
        "transcribed_text": text,
        "word_count": word_count,
        "language": "en"
    }
    
    return processed_data

async def save_results(processed_data: dict):
    """Step 3: Save the processed results"""
    # Access both the processed data and original transcription
    context = get_run_context()
    transcription = context.get_data("transcribe_audio.output")
    processed = context.get_data("process_text.output")
    
    # Save to database or file
    result = {
        "audio_file": processed["original_audio"],
        "transcription": transcription,
        "analysis": processed,
        "timestamp": "2025-01-15T14:30:25Z"
    }
    
    return f"Saved results for {result['audio_file']} with {result['analysis']['word_count']} words"

# Create and run the workflow
workflow = (Workflow(name='audio_processing')
    .step(transcribe_audio, audio_file_path="/path/to/audio.wav")
    .step(process_text, text="transcribe_audio.output")  # Pass previous step's output
    .step(save_results, processed_data="process_text.output")  # Pass previous step's output
)

# Run the workflow
result = await workflow.run()
print(result)`}/>

### Step-by-Step Variable Access

After running this workflow, each step has access to the following variables:

**Step `transcribe_audio`:**
<CodeBlock language="python" code={`.input.audio_file_path = '/path/to/audio.wav'
.output              = 'Transcribed audio from /path/to/audio.wav: Hello, this is a test recording.'
.audio_path          = '/path/to/audio.wav'`}/>

**Step `process_text`:**
<CodeBlock language="python" code={`.input.text                    = 'Transcribed audio from /path/to/audio.wav: Hello, this is a test recording.'
.output                        = {'original_audio': '/path/to/audio.wav', 'transcribed_text': '...', 'word_count': 8, 'language': 'en'}
transcribe_audio.output        = 'Transcribed audio from /path/to/audio.wav: Hello, this is a test recording.'
transcribe_audio.audio_path    = '/path/to/audio.wav'`}/>

**Step `save_results`:**
<CodeBlock language="python" code={`.input.processed_data           = {'original_audio': '/path/to/audio.wav', 'transcribed_text': '...', 'word_count': 8, 'language': 'en'}
.output                        = 'Saved results for /path/to/audio.wav with 8 words'
transcribe_audio.output        = 'Transcribed audio from /path/to/audio.wav: Hello, this is a test recording.'
process_text.output            = {'original_audio': '/path/to/audio.wav', 'transcribed_text': '...', 'word_count': 8, 'language': 'en'}`}/>

This example demonstrates how:
- **Input transformation**: Audio files are converted to text using `.input` parameters
- **Output chaining**: Each step's `.output` becomes the next step's `.input`
- **Cross-step access**: Steps can access data from any previous step using step names
- **Custom variables**: Additional context data is stored and shared using `set_data()` -->

<!-- ## Practical Use Cases

- **Data Pipeline**: Pass processed data between steps
- **State Management**: Track workflow progress and intermediate results  
- **Configuration**: Share settings across multiple steps
- **Error Handling**: Store error states and recovery information -->
