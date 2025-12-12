# Equality/Inequality
$eq: bool | number | string          # equals (shorter, matches MongoDB)
$ne: bool | number | string          # not equals

# Numeric comparisons
$gt: number                          # greater than
$gte: number                         # greater than or equal (more common than $ge)
$lt: number                          # less than
$lte: number                         # less than or equal (more common than $le)
$between: [number, number]           # inclusive range [min, max]
$multiple_of: number                 # divisible by
$positive: true                      # > 0
$negative: true                      # < 0
$non_negative: true                  # >= 0
$non_positive: true                  # <= 0

# String validators
$min_length: number
$max_length: number
$length: number                      # exact length
$starts_with: string
$ends_with: string
$contains: string
$not_contains: string
$pattern: string                     # regex pattern
$not_pattern: string                 # must NOT match regex
$matches: string                     # alias for $pattern
$email: true                         # valid email format
$url: true                           # valid URL format
$uuid: true                          # valid UUID format
$alpha: true                         # only letters
$alphanumeric: true                  # letters and numbers
$numeric: true                       # only numbers (as string)
$lowercase: true                     # all lowercase
$uppercase: true                     # all uppercase
$trim: string                        # after trimming, equals this
$one_of: [string, string, ...]       # in this list (clearer than $in for strings)

# Collection validators (arrays/strings)
$in: [value, value, ...]             # value is in this list
$not_in: [value, value, ...]         # value is NOT in this list
$empty: bool                         # is empty ([], "", etc)
$size: number                        # exact size/length (works for arrays and strings)
$min_size: number                    # minimum size
$max_size: number                    # maximum size

# Array-specific
$all: [{validators}]                 # all elements match these validators
$any: [{validators}]                 # at least one element matches
$none: [{validators}]                # no elements match
$unique: true                        # all elements are unique
$contains_any: [value, ...]          # contains at least one of these values
$contains_all: [value, ...]          # contains all of these values

# Object validators
$has_key: string | [string, ...]     # has this key(s)
$has_keys: [string, ...]             # alias for has_key with array
$not_has_key: string | [string, ...] # doesn't have this key(s)
$keys: [string, ...]                 # exact keys (no more, no less)
$required_keys: [string, ...]        # must have at least these keys
$empty: bool                         # is empty object {}

# Type validators
$type: "string" | "number" | "boolean" | "array" | "object" | "null"
$null: bool                          # is null
$not_null: bool                      # is not null
$exists: bool                        # key exists (even if null)
$defined: bool                       # not null and not undefined

# Logical operators
$and: [{validators}]                 # all conditions must pass
$or: [{validators}]                  # at least one condition must pass
$not: {validators}                   # condition must NOT pass
$nor: [{validators}]                 # none of the conditions pass

# Semantic/LLM validators
$semantic: string                    # semantically matches description
$not_semantic: string                # semantically does NOT match
$similar_to: string                  # semantically similar to this text
$sentiment: "positive" | "negative" | "neutral"
$language: "en" | "es" | "fr" ...    # detected language
$pii: bool                           # contains/doesn't contain PII Personally Identifiable Information
$toxic: bool                         # contains/doesn't contain toxic content

# Date/Time (if values are ISO strings)
$before: string                      # date before this
$after: string                       # date after this
$date_between: [string, string]      # date in range


## Examples

# Simple equality
output:
  $eq: "success"
  
# Numeric range
age:
  $gte: 18
  $lte: 65
  $not_in: [21, 22]
  
# String validation
email:
  $email: true
  $not_contains: "spam"
  $ends_with: ".com"
  
username:
  $min_length: 3
  $max_length: 20
  $alphanumeric: true
  $lowercase: true
  
# Array validation
tags:
  $min_size: 1
  $max_size: 10
  $unique: true
  $all:
    - $type: "string"
    - $not_empty: true
      
products:
  $contains_any: ["laptop", "phone"]
  $any:
    - price:
        $lt: 100
        
# Object validation
user:
  $required_keys: ["id", "email"]
  $not_has_key: "password"
  email:
    $email: true
    
# Logical combinations
status:
  $or:
    - $eq: "active"
    - $eq: "pending"
    - $starts_with: "temp_"
    
price:
  $and:
    - $gt: 0
    - $multiple_of: 0.01
    - $lte: 9999.99
    
# Semantic
description:
  $semantic: "a friendly product description"
  $not_semantic: "aggressive sales language"
  $min_length: 50
  $sentiment: "positive"
  $language: "en"
