# Brainstorming Evals V1

## Run Info <- Trace

- tokens (input, output, cached, ...)
- tool calls (total, per tool, error rate, elapsed time, ...)
- num. iterations
- time
- cost

## JSON Serializable types

- boolean
- number
- string
- array
- object
- null

## Validators

boolean:
  - ==

number:
  - ==
  - <=
  - >=
  - >
  - <
  - in []

string:
  - ==
  - length
  - startsWith
  - endsWith
  - contains
  - regex
  - in []
  - semantic
  
array:
  - *args

object:
  - **kwargs

all_of & any_of
