"""
 Copyright 2024 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import os


# Vertex AI Configuration
PROJECT_ID = os.getenv("GCP_PROJECT")
LOCATION = os.getenv("GCP_REGION")
MODEL_NAME = "gemini-1.5-flash-002"

# Generation Config
GENERATION_CONFIG = {
    "max_output_tokens": 8192,
    "temperature": 0.2,
    "top_p": 0.95
} 

ANALYSIS_CONFIGS = {
    "detailed": {
        "max_output_tokens": 8192,
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 40
    },
    "quick": {
        "max_output_tokens": 4096,
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 20
    },
    "validation": {
        "max_output_tokens": 2048,
        "temperature": 0.2,
        "top_p": 0.99,
        "top_k": 10
    }
}