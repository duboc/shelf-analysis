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


import google.generativeai as genai
from google.generativeai.types import GenerationConfig as GenaiGenerationConfig # Keep Part for direct creation if needed
# from google.generativeai.types import Part as GenaiPart # Not strictly needed if using from_bytes/from_uri
from config import PROJECT_ID, LOCATION, GCS_BUCKET_NAME # Added GCS_BUCKET_NAME

import json
import re
import logging
import os
import uuid
from google.cloud import storage

# Configure logger
logger = logging.getLogger(__name__)

def configure_genai():
    """Configures the google-genai SDK to use Vertex AI."""
    # The google-genai SDK uses environment variables GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION
    # when vertexai=True in genai.Client() or genai.configure() if not passed directly.
    # For Vertex AI, API key is not typically used directly in configure.
    # The environment should be authenticated (e.g., using gcloud auth application-default login).
    # Client and model instantiation will be in app.py.
    logger.info(f"google-genai will use Vertex AI with Project ID: {os.getenv('GOOGLE_CLOUD_PROJECT', PROJECT_ID)} and Location: {os.getenv('GOOGLE_CLOUD_LOCATION', LOCATION)}")
    # No explicit genai.configure() call is needed here if relying on env vars for Client.

def upload_to_gcs(file_bytes, original_file_name: str, content_type: str, bucket_name: str) -> str:
    """Uploads a file to Google Cloud Storage and returns its GCS URI.

    Args:
        file_bytes: The bytes of the file to upload.
        original_file_name: The original name of the file.
        content_type: The MIME type of the file (e.g., "video/mp4").
        bucket_name: The name of the GCS bucket.

    Returns:
        The GCS URI of the uploaded file (e.g., "gs://bucket_name/blob_name").
        
    Raises:
        ValueError: If the bucket_name is not provided.
        Exception: If the upload fails.
    """
    if not bucket_name:
        logger.error("GCS_BUCKET_NAME is not configured. Cannot upload file.")
        raise ValueError("GCS bucket name not provided or configured.")

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        file_extension = os.path.splitext(original_file_name)[1]
        # Ensure the blob name is unique and within GCS object naming guidelines
        # For example, remove problematic characters from original_file_name if included,
        # or just rely on UUID + extension.
        base_name = os.path.basename(original_file_name)
        safe_base_name = "".join(c if c.isalnum() or c in ['.', '_', '-'] else '_' for c in base_name)
        blob_name = f"uploads/{uuid.uuid4()}-{safe_base_name}"
        # If safe_base_name already includes an extension from original_file_name, 
        # and we also add file_extension, we might get "file.mp4.mp4".
        # Let's refine to ensure only one extension.
        if not safe_base_name.endswith(file_extension):
             blob_name = f"uploads/{uuid.uuid4()}-{os.path.splitext(safe_base_name)[0]}{file_extension}"
        else:
             blob_name = f"uploads/{uuid.uuid4()}-{safe_base_name}"


        blob = bucket.blob(blob_name)
        
        logger.info(f"Uploading {original_file_name} to gs://{bucket_name}/{blob_name} with content type {content_type}...")
        blob.upload_from_string(file_bytes, content_type=content_type)
        logger.info(f"File {original_file_name} uploaded successfully to gs://{bucket_name}/{blob_name}.")
        
        return f"gs://{bucket_name}/{blob_name}"
    except Exception as e:
        logger.error(f"Failed to upload {original_file_name} to GCS bucket {bucket_name}: {e}")
        raise

# def initialize_vertex_ai(project_id, location): # Removed
#     init(project=project_id, location=location)

def load_prompt(media_mime_type: str) -> str:
    """Loads the appropriate prompt based on the media type."""
    if media_mime_type and media_mime_type.startswith('video/'):
        prompt_file = 'prompt_video.md'
        logger.info("Loading video prompt from prompt_video.md")
    else:
        prompt_file = 'prompt.md'
        logger.info("Loading image prompt from prompt.md")
    
    try:
        with open(prompt_file, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        logger.error(f"Prompt file {prompt_file} not found.")
        # Return a default generic prompt or raise an error
        # For now, returning a generic prompt to avoid outright failure if a file is missing.
        # A more robust solution might involve raising an error or having a very basic default JSON structure.
        return "Analyze the provided media and describe its content in detail. Return response in JSON format."

def clean_json_response(text):
    # Remove markdown code block indicators
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Try to find JSON content between curly braces
    json_match = re.search(r'({[\s\S]*})', text)
    if json_match:
        return json_match.group(1)
    return text

# def analyze_shelf_image(model, image_data, prompt): # Commented out for now
#     # image_part = genai.types.Part.from_bytes(data=image_data, mime_type="image/jpeg") # Updated
    
#     # # Add explicit JSON request to the prompt
#     # full_prompt = (
#     #     f"{prompt}\n\n"
#     #     "Por favor, forneça a resposta APENAS no formato JSON especificado, "
#     #     "sem markdown ou texto adicional."
#     # )
    
#     # response = model.generate_content(
#     #     [image_part, full_prompt],
#     #     generation_config=GenaiGenerationConfig( # Updated
#     #         max_output_tokens=8192,
#     #         temperature=0.7,
#     #         top_p=0.95
#     #     )
#     # )
    
#     # # Clean and parse the response
#     # cleaned_response = clean_json_response(response.text)
#     # try:
#     #     json_response = json.loads(cleaned_response)
        
#     #     # Validate required structure
#     #     if 'analise_prateleira' not in json_response:
#     #         json_response = {'analise_prateleira': json_response}
            
#     #     return json_response
#     # except json.JSONDecodeError as e:
#     #     raise ValueError(f"Resposta inválida do modelo. Por favor, tente novamente.\nDetalhes: {str(e)}")