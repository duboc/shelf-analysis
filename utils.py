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


from vertexai import init
from vertexai.generative_models import GenerativeModel, GenerationConfig, Part
import json
import re

def initialize_vertex_ai(project_id, location):
    init(project=project_id, location=location)

def load_prompt():
    with open('prompt.md', 'r', encoding='utf-8') as file:
        return file.read()

def clean_json_response(text):
    # Remove markdown code block indicators
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Try to find JSON content between curly braces
    json_match = re.search(r'({[\s\S]*})', text)
    if json_match:
        return json_match.group(1)
    return text

def analyze_shelf_image(model, image_data, prompt):
    image_part = Part.from_data(image_data, mime_type="image/jpeg")
    
    # Add explicit JSON request to the prompt
    full_prompt = (
        f"{prompt}\n\n"
        "Por favor, forneça a resposta APENAS no formato JSON especificado, "
        "sem markdown ou texto adicional."
    )
    
    response = model.generate_content(
        [image_part, full_prompt],
        generation_config=GenerationConfig(
            max_output_tokens=8192,
            temperature=0.7,  # Reduced temperature for more consistent output
            top_p=0.95
        )
    )
    
    # Clean and parse the response
    cleaned_response = clean_json_response(response.text)
    try:
        json_response = json.loads(cleaned_response)
        
        # Validate required structure
        if 'analise_prateleira' not in json_response:
            json_response = {'analise_prateleira': json_response}
            
        return json_response
    except json.JSONDecodeError as e:
        raise ValueError(f"Resposta inválida do modelo. Por favor, tente novamente.\nDetalhes: {str(e)}") 