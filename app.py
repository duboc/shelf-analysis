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

import streamlit as st
import pandas as pd
import plotly.express as px
from config import PROJECT_ID, LOCATION, MODEL_NAME
from utils import initialize_vertex_ai, load_prompt, clean_json_response
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
import json
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from PIL import Image
import io
import numpy as np
import os

st.set_page_config(page_title="Análise de Prateleira", page_icon=":bar_chart:", layout="wide")

class StreamlitLogHandler(logging.Handler):
    def __init__(self, st_container):
        super().__init__()
        self.st_container = st_container
        # Initialize session state for logs if it doesn't exist
        if 'logs' not in st.session_state:
            st.session_state.logs = []
        
    def emit(self, record):
        try:
            log_entry = self.format(record)
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Add new log entry to session state
            st.session_state.logs.append(f"[{timestamp}] {log_entry}")
            
            # Keep only the last 100 logs
            if len(st.session_state.logs) > 100:
                st.session_state.logs = st.session_state.logs[-100:]
            
            # Display logs
            self.st_container.code("\n".join(st.session_state.logs))
            
        except Exception as e:
            print(f"Error in StreamlitLogHandler: {str(e)}")

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass
class ValidationThresholds:
    """Configuration for validation thresholds."""
    min_brand_confidence: float = 70.0
    min_count_confidence: float = 80.0
    min_overall_confidence: float = 75.0
    max_count_discrepancy: float = 0.2  # 20% threshold
    max_share_discrepancy: float = 0.15  # 15% threshold
    min_image_clarity: float = 50.0
    min_shelf_visibility: float = 70.0
    min_logo_visibility: float = 60.0

def analyze_shelf_image_pipeline(model, image_data):
    # Stage 1: Initial product detection and counting
    product_analysis = analyze_products(model, image_data)
    
    # Stage 2: Brand and manufacturer identification
    manufacturer_analysis = analyze_manufacturers(model, image_data, product_analysis)
    
    # Stage 3: Shelf organization analysis
    organization_analysis = analyze_shelf_organization(model, image_data)
    
    # Stage 4: Competitive analysis
    competitive_analysis = analyze_competitive_landscape(model, image_data, manufacturer_analysis)
    
    # Stage 5: Validation and cross-checking
    validated_results = validate_analysis(model, [
        product_analysis,
        manufacturer_analysis,
        organization_analysis,
        competitive_analysis
    ])
    
    return validated_results

def save_response(response_data, filename):
    """Save response data to a local file."""
    os.makedirs('responses', exist_ok=True)
    filepath = os.path.join('responses', filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(response_data, f, ensure_ascii=False, indent=2)
    return filepath

def load_response(filename):
    """Load response data from a local file."""
    filepath = os.path.join('responses', filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def analyze_shelf_image_structured(model, image_data, prompt):
    # Add validation thresholds configuration
    thresholds = ValidationThresholds(
        min_brand_confidence=75.0,
        min_count_confidence=85.0,
        min_overall_confidence=80.0,
        max_count_discrepancy=0.15,
        max_share_discrepancy=0.1,
        min_image_clarity=60.0,
        min_shelf_visibility=75.0,
        min_logo_visibility=65.0
    )
    
    # Add image quality validation
    logger.info("Validating image quality")
    try:
        quality_results = validate_image_quality(model, image_data)
        save_response(quality_results, f'quality_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        
        image_clarity = quality_results.get('image_clarity', 100)
        shelf_visibility = quality_results.get('shelf_visibility', 100)
        
        if image_clarity < 50:
            logger.warning(f"Low image clarity: {image_clarity}/100")
            st.warning("A qualidade da imagem está baixa, os resultados podem ser imprecisos.")
        
        if shelf_visibility < 70:
            logger.warning(f"Low shelf visibility: {shelf_visibility}%")
            st.warning("A visibilidade da prateleira está baixa, os resultados podem ser imprecisos.")
        
    except Exception as e:
        logger.warning(f"Image quality validation failed: {str(e)}")
        st.warning("Não foi possível validar a qualidade da imagem, prosseguindo com a análise.")
    
    # Generation configs for different stages
    analysis_config = GenerationConfig(
        max_output_tokens=8192,
        temperature=0.4,
        top_p=0.95,
        top_k=40,
        seed=42
    )
    
    # Create a sample response structure based on the schema
    example_response = {
        "analise_prateleira": {
            "fabricantes": [
                {
                    "nome": "Exemplo Fabricante",
                    "contagem": 5,
                    "visual_shelf_share": 25.5,
                    "posicao_prateleira": "meio",
                    "visibilidade": 85,
                    "confianca": 95
                }
            ],
            "meta": {
                "total_produtos": 0,
                "total_fabricantes": 0,
                "confianca_media": 0,
                "timestamp": ""
            }
        }
    }
    
    # Create the structured prompt
    structured_prompt = f"""
    {prompt}
    
    Analise a imagem e retorne um objeto JSON com a seguinte estrutura:
    {json.dumps(example_response, indent=2, ensure_ascii=False)}
    
    Regras importantes:
    1. A soma total do visual_shelf_share deve ser 100%
    2. Contagem deve refletir produtos visíveis
    3. Confiança deve ser entre 0-100
    4. Retorne APENAS o JSON, sem texto adicional
    """
    
    try:
        # Get initial analysis
        logger.info("Performing initial analysis")
        response = model.generate_content(
            [Part.from_data(image_data, "image/jpeg"), structured_prompt],
            generation_config=analysis_config,
            stream=False
        )
        
        # Clean and parse response
        cleaned_response = clean_json_response(response.text)
        if not cleaned_response:
            raise ValueError("Empty response from model")
        
        result = json.loads(cleaned_response)
        validate_product_data(result)
        
        # Save response to file
        save_response(result, f'analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        raise ValueError(f"Error processing JSON: {str(e)}")
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise

def validate_image_quality(model, image_data):
    """Validate image quality and suitability for analysis."""
    quality_prompt = """
    Analyze the image quality and return a JSON with the following metrics:
    {
        "image_clarity": 0-100 score,
        "lighting_conditions": "poor/adequate/good",
        "shelf_visibility": 0-100 percentage,
        "image_angle": "straight/angled",
        "obstructions": ["list", "of", "issues"]
    }
    Return ONLY the JSON object, no additional text.
    """
    try:
        response = model.generate_content([
            Part.from_data(image_data, "image/jpeg"),
            quality_prompt
        ])
        
        # Clean and parse the response
        quality_data = clean_json_response(response.text)
        if not quality_data:
            logger.warning("Could not parse image quality response")
            return {"image_clarity": 100, "shelf_visibility": 100}
            
        try:
            quality_result = json.loads(quality_data)
            return quality_result
        except json.JSONDecodeError:
            logger.warning("Invalid JSON in image quality response")
            return {"image_clarity": 100, "shelf_visibility": 100}
            
    except Exception as e:
        logger.warning(f"Image quality check failed: {str(e)}")
        return {"image_clarity": 100, "shelf_visibility": 100}

def analyze_shelf_image_recommendations(model, image_data):
    generation_config = GenerationConfig(
        max_output_tokens=8192,
        temperature=0.2,
        top_p=0.95
    )
    
    recommendations_prompt = """Analise a imagem da prateleira e forneça:
    1. Observações detalhadas sobre a disposição dos produtos
    2. Considerações estratégicas para melhorar o visual merchandising
    3. Recomendações práticas para otimização do espaço
    
    Por favor, forneça uma análise detalhada e profissional."""
    
    response = model.generate_content(
        [recommendations_prompt, Part.from_data(image_data, "image/jpeg")],
        generation_config=generation_config,
        stream=False
    )
    
    # Save recommendations to file
    recommendations = response.text
    save_response({"recommendations": recommendations}, 
                 f'recommendations_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    
    return recommendations

def create_visualizations(data):
    try:
        # Ensure the data structure exists
        if 'analise_prateleira' not in data or 'fabricantes' not in data['analise_prateleira']:
            raise ValueError("Estrutura de dados inválida na resposta")
            
        # Extract fabricantes data into a DataFrame
        df = pd.DataFrame(data['analise_prateleira']['fabricantes'])
        
        if df.empty:
            raise ValueError("Nenhum dado de fabricante encontrado")
        
        # Print column names for debugging
        print("Available columns:", df.columns.tolist())
            
        # Rename columns to match our expected names
        column_mapping = {
            'contagem': 'contagem_remedios',
            'shelf_share': 'visual_shelf_share',
            'visualShelfShare': 'visual_shelf_share'
        }
        
        # Only rename columns that exist
        existing_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=existing_columns)
        
        # Sort DataFrame by shelf_share for better visualization
        df = df.sort_values(by='visual_shelf_share', ascending=False)
        
        # Calculate additional metrics
        df['market_dominance'] = df['visual_shelf_share'] * df['contagem_remedios'] / 100
        df['avg_space_per_product'] = df['visual_shelf_share'] / df['contagem_remedios']
        
        # 1. Enhanced Bar Charts
        fig_shelf_share = px.bar(
            df,
            x='nome',
            y='visual_shelf_share',
            title='Visual Shelf Share por Fabricante',
            labels={'nome': 'Fabricante', 'visual_shelf_share': 'Visual Shelf Share (%)'},
            color='visual_shelf_share',
            color_continuous_scale='Viridis'
        )
        
        fig_product_count = px.bar(
            df,
            x='nome',
            y='contagem_remedios',
            title='Quantidade de Produtos por Fabricante',
            labels={'nome': 'Fabricante', 'contagem_remedios': 'Número de Produtos'},
            color='contagem_remedios',
            color_continuous_scale='Viridis'
        )
        
        # 2. Pie and Donut Charts
        fig_pie = px.pie(
            df,
            values='visual_shelf_share',
            names='nome',
            title='Distribuição do Visual Shelf Share',
            hole=0.0  # Set to 0 for pie, 0.6 for donut
        )
        
        fig_donut = px.pie(
            df,
            values='contagem_remedios',
            names='nome',
            title='Distribuição de Produtos',
            hole=0.6
        )
        
        # 3. Treemap for Hierarchical View
        fig_treemap = px.treemap(
            df,
            path=['nome'],
            values='contagem_remedios',
            color='visual_shelf_share',
            title='Treemap: Produtos e Shelf Share',
            color_continuous_scale='RdYlBu'
        )
        
        # 4. Scatter Plot with Size
        fig_scatter = px.scatter(
            df,
            x='contagem_remedios',
            y='visual_shelf_share',
            size='market_dominance',
            color='avg_space_per_product',
            text='nome',
            title='Análise Multidimensional de Mercado',
            labels={
                'contagem_remedios': 'Número de Produtos',
                'visual_shelf_share': 'Visual Shelf Share (%)',
                'market_dominance': 'Dominância de Mercado',
                'avg_space_per_product': 'Espaço Médio por Produto'
            }
        )
        
        # 5. Radar Chart for Brand Comparison
        fig_radar = px.line_polar(
            df,
            r='visual_shelf_share',
            theta='nome',
            line_close=True,
            title='Comparação Radial de Marcas'
        )
        
        # 6. Funnel Chart for Market Position
        df_sorted = df.sort_values('market_dominance', ascending=True)
        fig_funnel = px.funnel(
            df_sorted,
            x='market_dominance',
            y='nome',
            title='Funil de Dominância de Mercado'
        )
        
        # 7. Heatmap of Metrics
        correlation_data = df[['contagem_remedios', 'visual_shelf_share', 'market_dominance', 'avg_space_per_product']].corr()
        fig_heatmap = px.imshow(
            correlation_data,
            title='Correlação entre Métricas',
            color_continuous_scale='RdBu'
        )
        
        # Customize the layout of all charts
        charts = [fig_shelf_share, fig_product_count, fig_scatter, fig_pie, 
                 fig_donut, fig_treemap, fig_radar, fig_funnel, fig_heatmap]
        
        for fig in charts:
            fig.update_layout(
                height=400,
                margin=dict(l=50, r=50, t=50, b=50),
                showlegend=True,
                template='plotly_white'
            )
        
        return (fig_shelf_share, fig_product_count, fig_pie, fig_donut, 
                fig_treemap, fig_scatter, fig_radar, fig_funnel, fig_heatmap, 
                df, 'contagem_remedios', 'visual_shelf_share')
        
    except Exception as e:
        raise ValueError(f"Erro ao criar visualizações: {str(e)}\nColunas disponíveis: {df.columns.tolist()}")

def analyze_products(model, image_data):
    product_prompt = """
    Focus only on identifying and counting individual products:
    1. Count total number of visible products
    2. Identify product types (boxes, bottles, etc.)
    3. Note any partially visible products
    Return results in JSON format.
    """
    response = model.generate_content([
        Part.from_data(image_data, "image/jpeg"),
        product_prompt
    ])
    return clean_json_response(response.text)

def analyze_manufacturers(model, image_data, product_analysis):
    manufacturer_prompt = f"""
    Based on the previous product analysis: {product_analysis}
    Focus on manufacturer identification:
    1. List all visible brands
    2. Calculate shelf share per manufacturer
    3. Identify premium shelf positions
    Return results in JSON format.
    """
    response = model.generate_content([
        Part.from_data(image_data, "image/jpeg"),
        manufacturer_prompt
    ])
    return clean_json_response(response.text)

def validate_analysis(model, analysis_results):
    validation_prompt = f"""
    Review and validate the following analysis results:
    {json.dumps(analysis_results, indent=2)}
    
    Check for:
    1. Consistency in product counts
    2. Total shelf share adds up to 100%
    3. Manufacturer presence validation
    4. Logical arrangement patterns
    
    Return validated and corrected results in JSON format.
    """
    
    response = model.generate_content(
        validation_prompt,
        generation_config=GenerationConfig(temperature=0.2)
    )
    return clean_json_response(response.text)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def robust_analysis(model, image_data, analysis_type, config):
    try:
        response = model.generate_content(
            [Part.from_data(image_data, "image/jpeg"), analysis_type],
            generation_config=config
        )
        return validate_response(response)
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

async def parallel_analysis(model, image_data):
    tasks = [
        asyncio.create_task(analyze_products(model, image_data)),
        asyncio.create_task(analyze_shelf_organization(model, image_data)),
        asyncio.create_task(analyze_competitive_landscape(model, image_data))
    ]
    results = await asyncio.gather(*tasks)
    return combine_results(results)

def analyze_shelf_organization(model, image_data):
    organization_prompt = """
    Focus on shelf organization patterns:
    1. Identify product placement patterns
    2. Note vertical and horizontal organization
    3. Identify premium shelf positions and their usage
    4. Analyze spacing and product alignment
    Return results in JSON format.
    """
    response = model.generate_content([
        Part.from_data(image_data, "image/jpeg"),
        organization_prompt
    ])
    return clean_json_response(response.text)

def analyze_competitive_landscape(model, image_data, manufacturer_analysis=None):
    competitive_prompt = """
    Analyze competitive positioning:
    1. Compare brand presence and positioning
    2. Identify dominant brands and their strategies
    3. Note any competitive advantages in placement
    4. Analyze price point positioning if visible
    Return results in JSON format.
    """
    response = model.generate_content([
        Part.from_data(image_data, "image/jpeg"),
        competitive_prompt
    ])
    return clean_json_response(response.text)

def validate_response(response):
    """Validate and clean the model's response."""
    if not response or not response.text:
        raise ValueError("Empty response from model")
    try:
        # Try to parse as JSON if applicable
        return json.loads(response.text)
    except json.JSONDecodeError:
        # If not JSON, return raw text
        return response.text

def combine_results(results):
    """Combine parallel analysis results into a single coherent output."""
    combined = {"analise_prateleira": {"fabricantes": []}}
    manufacturers = {}
    
    # Collect all manufacturer data
    for result in results:
        if not isinstance(result, dict) or 'analise_prateleira' not in result:
            continue
            
        for mfr in result['analise_prateleira'].get('fabricantes', []):
            name = mfr['nome']
            if name not in manufacturers:
                manufacturers[name] = []
            manufacturers[name].append(mfr)
    
    # Average the values for each manufacturer
    for name, data_list in manufacturers.items():
        avg_count = sum(m['contagem'] for m in data_list) / len(data_list)
        avg_share = sum(m['visual_shelf_share'] for m in data_list) / len(data_list)
        
        combined['analise_prateleira']['fabricantes'].append({
            "nome": name,
            "contagem": round(avg_count),
            "visual_shelf_share": round(avg_share, 1)
        })
    
    return combined

def normalize_shelf_share(manufacturers):
    """Normalize shelf share values to sum to 100%."""
    total_share = sum(m['visual_shelf_share'] for m in manufacturers)
    if total_share == 0:
        return manufacturers
    
    # Normalize each value
    for manufacturer in manufacturers:
        manufacturer['visual_shelf_share'] = (manufacturer['visual_shelf_share'] / total_share) * 100
        # Round to 1 decimal place
        manufacturer['visual_shelf_share'] = round(manufacturer['visual_shelf_share'], 1)
    
    # Adjust rounding errors to ensure exact 100% total
    total_after = sum(m['visual_shelf_share'] for m in manufacturers)
    if total_after != 100:
        # Add the difference to the largest share to reach exactly 100%
        diff = 100 - total_after
        max_share_manufacturer = max(manufacturers, key=lambda x: x['visual_shelf_share'])
        max_share_manufacturer['visual_shelf_share'] += diff
        max_share_manufacturer['visual_shelf_share'] = round(max_share_manufacturer['visual_shelf_share'], 1)
    
    return manufacturers

def validate_product_data(data):
    """Validate product analysis data structure and values."""
    required_fields = ['nome', 'contagem', 'visual_shelf_share']
    
    if not isinstance(data, dict):
        raise ValueError("Invalid data structure: expected dictionary")
        
    if 'analise_prateleira' not in data:
        raise ValueError("Missing 'analise_prateleira' key in response")
        
    if 'fabricantes' not in data['analise_prateleira']:
        raise ValueError("Missing 'fabricantes' key in response")
        
    manufacturers = data['analise_prateleira']['fabricantes']
    if not isinstance(manufacturers, list):
        raise ValueError("'fabricantes' must be a list")
        
    for manufacturer in manufacturers:
        missing_fields = [field for field in required_fields if field not in manufacturer]
        if missing_fields:
            raise ValueError(f"Missing required fields for manufacturer: {missing_fields}")
            
        if not isinstance(manufacturer['contagem'], (int, float)):
            raise ValueError("Product count must be a number")
            
        if not isinstance(manufacturer['visual_shelf_share'], (int, float)):
            raise ValueError("Shelf share must be a number")
            
        if manufacturer['visual_shelf_share'] < 0:
            raise ValueError("Shelf share must be positive")
    
    # Calculate total shelf share
    total_share = sum(m['visual_shelf_share'] for m in manufacturers)
    
    # If total is not approximately 100%, normalize the values
    if not (95 <= total_share <= 105):
        logger.warning(f"Total shelf share ({total_share}%) needs normalization")
        data['analise_prateleira']['fabricantes'] = normalize_shelf_share(manufacturers)
        logger.info("Shelf share values normalized to 100%")
    
    return True

def cross_validate_analysis(model, image_data, initial_result, thresholds: Optional[ValidationThresholds] = None):
    """Cross-validate results with a second analysis."""
    if thresholds is None:
        thresholds = ValidationThresholds()
        
    cross_validation_prompt = f"""
    Validate the following shelf analysis results and identify any discrepancies:
    {json.dumps(initial_result, indent=2)}
    
    Return a JSON with EXACTLY this structure:
    {{
        "validations": [
            {{
                "manufacturer": "name",
                "confidence_score": 0-100,
                "confirmed": true/false,
                "suggested_correction": {{
                    "contagem": number,
                    "visual_shelf_share": number
                }},
                "validation_details": {{
                    "brand_confidence": 0-100,
                    "count_confidence": 0-100,
                    "share_confidence": 0-100,
                    "position_confidence": 0-100,
                    "validation_method": "string"
                }}
            }}
        ],
        "discrepancies": [
            {{
                "type": "description",
                "severity": "high/medium/low",
                "details": "explanation",
                "confidence": 0-100,
                "impact_score": 0-100,
                "suggested_resolution": "string"
            }}
        ],
        "meta": {{
            "validation_timestamp": "string",
            "overall_confidence": 0-100,
            "validation_coverage": 0-100,
            "quality_metrics": {{
                "image_quality_score": 0-100,
                "detection_confidence": 0-100,
                "validation_reliability": 0-100
            }}
        }}
    }}
    """
    
    try:
        response = model.generate_content(
            [Part.from_data(image_data, "image/jpeg"), cross_validation_prompt],
            generation_config=GenerationConfig(
                temperature=0.2,
                seed=321,  # Fixed seed for cross-validation
                top_k=10,
                top_p=0.95
            )
        )
        
        validation_text = clean_json_response(response.text)
        if not validation_text:
            logger.warning("Empty validation response")
            return create_default_validation_response()
            
        try:
            validation_result = json.loads(validation_text)
            
            # Validate against thresholds
            validation_result = apply_validation_thresholds(validation_result, thresholds)
            
            # Add validation metadata
            validation_result['meta'] = {
                **validation_result.get('meta', {}),
                'validation_timestamp': datetime.now().isoformat(),
                'thresholds_applied': asdict(thresholds)
            }
            
            return validation_result
        except json.JSONDecodeError:
            logger.warning("Invalid JSON in validation response")
            return create_default_validation_response()
            
    except Exception as e:
        logger.warning(f"Cross-validation failed: {str(e)}")
        return create_default_validation_response()

def create_default_validation_response():
    """Create a default validation response structure."""
    return {
        "validations": [],
        "discrepancies": [],
        "meta": {
            "validation_timestamp": datetime.now().isoformat(),
            "overall_confidence": 0,
            "validation_coverage": 0,
            "quality_metrics": {
                "image_quality_score": 0,
                "detection_confidence": 0,
                "validation_reliability": 0
            }
        }
    }

def apply_validation_thresholds(validation_result: Dict, thresholds: ValidationThresholds) -> Dict:
    """Apply validation thresholds and add appropriate discrepancies."""
    new_discrepancies = []
    
    # Check validations against thresholds
    for validation in validation_result.get('validations', []):
        details = validation.get('validation_details', {})
        
        # Check brand confidence
        if details.get('brand_confidence', 0) < thresholds.min_brand_confidence:
            new_discrepancies.append({
                "type": "low_brand_confidence",
                "severity": "high",
                "details": f"Brand confidence below threshold for {validation['manufacturer']}",
                "confidence": 95,
                "impact_score": 80,
                "suggested_resolution": "Manual brand verification required"
            })
        
        # Check count confidence
        if details.get('count_confidence', 0) < thresholds.min_count_confidence:
            new_discrepancies.append({
                "type": "low_count_confidence",
                "severity": "medium",
                "details": f"Count confidence below threshold for {validation['manufacturer']}",
                "confidence": 90,
                "impact_score": 70,
                "suggested_resolution": "Perform manual count verification"
            })
    
    # Add new discrepancies to the result
    validation_result['discrepancies'] = [
        *validation_result.get('discrepancies', []),
        *new_discrepancies
    ]
    
    # Update meta information
    meta = validation_result.get('meta', {})
    quality_metrics = meta.get('quality_metrics', {})
    
    if quality_metrics.get('image_quality_score', 0) < thresholds.min_image_clarity:
        new_discrepancies.append({
            "type": "low_image_quality",
            "severity": "high",
            "details": "Image quality below acceptable threshold",
            "confidence": 100,
            "impact_score": 90,
            "suggested_resolution": "Request higher quality image"
        })
    
    # Update overall confidence based on thresholds
    if meta.get('overall_confidence', 0) < thresholds.min_overall_confidence:
        new_discrepancies.append({
            "type": "low_overall_confidence",
            "severity": "high",
            "details": "Overall confidence below acceptable threshold",
            "confidence": 100,
            "impact_score": 85,
            "suggested_resolution": "Consider re-analysis with better image or manual verification"
        })
    
    return validation_result

def combine_analysis_results(results_list):
    """Combine multiple analysis results into a single consensus result with confidence scores."""
    combined = {
        "analise_prateleira": {
            "fabricantes": [],
            "meta": {
                "total_produtos": 0,
                "total_fabricantes": 0,
                "confianca_media": 0,
                "timestamp": datetime.now().isoformat()
            }
        }
    }
    manufacturers = {}
    
    # Collect all manufacturer data with confidence scores
    for result in results_list:
        if not isinstance(result, dict) or 'analise_prateleira' not in result:
            continue
            
        for mfr in result['analise_prateleira'].get('fabricantes', []):
            name = mfr['nome']
            if name not in manufacturers:
                manufacturers[name] = []
            manufacturers[name].append({
                "contagem": mfr['contagem'],
                "visual_shelf_share": mfr['visual_shelf_share'],
                "confianca": mfr.get('confianca', 85)  # Default confidence if not provided
            })
    
    # Calculate weighted averages based on confidence scores
    for name, data_list in manufacturers.items():
        total_weight = sum(d['confianca'] for d in data_list)
        if total_weight == 0:
            continue
            
        avg_count = sum(d['contagem'] * d['confianca'] for d in data_list) / total_weight
        avg_share = sum(d['visual_shelf_share'] * d['confianca'] for d in data_list) / total_weight
        avg_confidence = sum(d['confianca'] for d in data_list) / len(data_list)
        
        combined['analise_prateleira']['fabricantes'].append({
            "nome": name,
            "contagem": round(avg_count),
            "visual_shelf_share": round(avg_share, 1),
            "confianca": round(avg_confidence, 1)
        })
    
    # Update meta information
    meta = combined['analise_prateleira']['meta']
    meta['total_produtos'] = sum(m['contagem'] for m in combined['analise_prateleira']['fabricantes'])
    meta['total_fabricantes'] = len(combined['analise_prateleira']['fabricantes'])
    meta['confianca_media'] = round(
        sum(m['confianca'] for m in combined['analise_prateleira']['fabricantes']) / 
        len(combined['analise_prateleira']['fabricantes'])
        if combined['analise_prateleira']['fabricantes'] else 0,
        1
    )
    
    return combined

def crop_image_to_quadrants(image_data: bytes, grid_size: int = 4) -> List[Tuple[bytes, Tuple[int, int]]]:
    """
    Crop the image into a grid of quadrants.
    
    Args:
        image_data: The original image data in bytes
        grid_size: The size of the grid (e.g., 4 for a 4x4 grid resulting in 16 quadrants)
        
    Returns:
        List of tuples containing (cropped_image_bytes, (row_index, col_index))
    """
    # Open the image using PIL
    image = Image.open(io.BytesIO(image_data))
    
    # Get image dimensions
    width, height = image.size
    
    # Calculate quadrant dimensions
    quad_width = width // grid_size
    quad_height = height // grid_size
    
    quadrants = []
    
    # Crop image into quadrants
    for row in range(grid_size):
        for col in range(grid_size):
            # Calculate coordinates for cropping
            left = col * quad_width
            top = row * quad_height
            right = left + quad_width
            bottom = top + quad_height
            
            # Crop the quadrant
            quadrant = image.crop((left, top, right, bottom))
            
            # Convert quadrant to bytes
            quadrant_bytes = io.BytesIO()
            quadrant.save(quadrant_bytes, format=image.format or 'JPEG')
            quadrant_bytes = quadrant_bytes.getvalue()
            
            quadrants.append((quadrant_bytes, (row, col)))
    
    return quadrants

def combine_quadrant_results(results: List[Tuple[Dict, Tuple[int, int]]]) -> Dict:
    """Combine results from parallel quadrant analysis."""
    combined = {
        "analise_prateleira": {
            "fabricantes": [],
            "quadrants": {},
            "meta": {
                "total_produtos": 0,
                "total_fabricantes": 0,
                "confianca_media": 0,
                "timestamp": datetime.now().isoformat(),
                "grid_size": int(len(results) ** 0.5)
            }
        }
    }
    
    manufacturers_data = {}
    
    # Process results from each quadrant
    for result, position in results:
        if result and 'analise_prateleira' in result:
            quadrant_key = f"{position[0]},{position[1]}"
            combined['analise_prateleira']['quadrants'][quadrant_key] = result
            
            # Aggregate manufacturer data
            for mfr in result['analise_prateleira'].get('fabricantes', []):
                name = mfr['nome']
                if name not in manufacturers_data:
                    manufacturers_data[name] = {
                        'contagem': 0,
                        'visual_shelf_share': 0,
                        'quadrant_appearances': 0,
                        'confidence_sum': 0
                    }
                
                manufacturers_data[name]['contagem'] += mfr['contagem']
                manufacturers_data[name]['visual_shelf_share'] += mfr['visual_shelf_share']
                manufacturers_data[name]['quadrant_appearances'] += 1
                manufacturers_data[name]['confidence_sum'] += mfr.get('confianca', 85)
    
    # Calculate final manufacturer statistics
    for name, data in manufacturers_data.items():
        appearances = data['quadrant_appearances']
        combined['analise_prateleira']['fabricantes'].append({
            "nome": name,
            "contagem": data['contagem'],
            "visual_shelf_share": round(data['visual_shelf_share'] / appearances, 1),
            "confianca": round(data['confidence_sum'] / appearances, 1),
            "quadrant_presence": appearances
        })
    
    # Normalize shelf share to 100%
    total_share = sum(mfr['visual_shelf_share'] for mfr in combined['analise_prateleira']['fabricantes'])
    if total_share > 0:
        for mfr in combined['analise_prateleira']['fabricantes']:
            mfr['visual_shelf_share'] = round((mfr['visual_shelf_share'] / total_share) * 100, 1)
    
    return combined

def parallel_quadrant_analysis(model, quadrants: List[Tuple[bytes, Tuple[int, int]]], prompt: str) -> Dict:
    """
    Perform parallel analysis of quadrants using the flash model.
    """
    def process_single_quadrant(quadrant_data: bytes, position: Tuple[int, int]):
        try:
            logger.info(f"Analyzing quadrant at position {position}")
            result = analyze_shelf_image_structured(model, quadrant_data, prompt)
            # Save quadrant result
            save_response(result, f'quadrant_{position[0]}_{position[1]}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            return result, position
        except Exception as e:
            logger.error(f"Error analyzing quadrant at position {position}: {str(e)}")
            return None, position

    # Process quadrants in batches
    batch_size = 4
    all_results = []
    
    for i in range(0, len(quadrants), batch_size):
        batch = quadrants[i:i + batch_size]
        batch_results = []
        
        for quadrant_data, position in batch:
            result = process_single_quadrant(quadrant_data, position)
            batch_results.append(result)
            
        all_results.extend(batch_results)

    # Combine results
    final_results = combine_quadrant_results(all_results)
    save_response(final_results, f'quadrants_combined_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    
    return final_results

def analyze_shelf(image_data, model, prompt, tab1, tab2, tab3, tab4, grid_size):
    """Function to handle the shelf analysis process."""
    logger.info("Iniciando análise de nova imagem")
    
    with st.spinner('Analisando a imagem...'):
        try:
            logger.info("Imagem carregada com sucesso")
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Crop image into quadrants
            status_text.text("Dividindo imagem em quadrantes...")
            quadrants = crop_image_to_quadrants(image_data, grid_size)
            progress_bar.progress(10)
            
            # Display quadrants in the Quadrants tab
            with tab4:
                st.subheader('Visualização dos Quadrantes')
                for i in range(grid_size):
                    cols = st.columns(grid_size)
                    for j in range(grid_size):
                        with cols[j]:
                            quadrant_data, _ = quadrants[i * grid_size + j]
                            st.image(quadrant_data, caption=f"Quadrante ({i},{j})", use_column_width=True)
            
            # Analyze quadrants
            status_text.text("Analisando quadrantes...")
            quadrant_result = parallel_quadrant_analysis(model, quadrants, prompt)
            progress_bar.progress(40)
            
            # Analyze full image
            status_text.text("Analisando imagem completa...")
            full_result = analyze_shelf_image_structured(model, image_data, prompt)
            progress_bar.progress(70)
            
            # Combine results
            status_text.text("Combinando resultados...")
            final_result = combine_analysis_results([full_result, quadrant_result])
            save_response(final_result, f'final_result_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            progress_bar.progress(90)
            
            # Get recommendations
            status_text.text("Gerando recomendações...")
            recommendations = analyze_shelf_image_recommendations(model, image_data)
            progress_bar.progress(100)
            status_text.text("Análise concluída!")
            
            # Display results in tabs
            with tab3:
                st.subheader('JSON Raw')
                st.json(final_result)
                
                # Display quadrant analysis
                st.subheader('Análise por Quadrantes')
                if 'quadrants' in final_result['analise_prateleira']:
                    for i in range(grid_size):
                        cols = st.columns(grid_size)
                        for j in range(grid_size):
                            quadrant_key = f"{i},{j}"
                            with cols[j]:
                                if quadrant_key in final_result['analise_prateleira']['quadrants']:
                                    st.write(f"Quadrant {quadrant_key}")
                                    st.json(final_result['analise_prateleira']['quadrants'][quadrant_key])
            
            with tab2:
                st.subheader('Análise e Recomendações')
                st.write(recommendations)
            
            with tab1:
                st.subheader('Análise Visual')
                try:
                    logger.info("Gerando visualizações")
                    (fig_shelf_share, fig_product_count, fig_pie, fig_donut, 
                     fig_treemap, fig_scatter, fig_radar, fig_funnel, fig_heatmap,
                     df, count_column, shelf_share_column) = create_visualizations(final_result)
                    
                    # Create metric cards at the top
                    metric_cols = st.columns(4)
                    with metric_cols[0]:
                        st.metric("Total de Produtos", f"{df[count_column].sum():.0f}")
                    with metric_cols[1]:
                        st.metric("Total de Fabricantes", f"{len(df):.0f}")
                    with metric_cols[2]:
                        st.metric("Média de Shelf Share", f"{df[shelf_share_column].mean():.1f}%")
                    with metric_cols[3]:
                        st.metric("Maior Share", f"{df[shelf_share_column].max():.1f}%")
                    
                    # Create tabs for different chart categories
                    viz_tabs = st.tabs(["Distribuição", "Comparação", "Correlação", "Avançado"])
                    
                    with viz_tabs[0]:  # Distribution Charts
                        cols1 = st.columns(2)
                        with cols1[0]:
                            st.plotly_chart(fig_pie, use_container_width=True)
                        with cols1[1]:
                            st.plotly_chart(fig_donut, use_container_width=True)
                        st.plotly_chart(fig_treemap, use_container_width=True)
                    
                    with viz_tabs[1]:  # Comparison Charts
                        cols2 = st.columns(2)
                        with cols2[0]:
                            st.plotly_chart(fig_shelf_share, use_container_width=True)
                        with cols2[1]:
                            st.plotly_chart(fig_product_count, use_container_width=True)
                        st.plotly_chart(fig_radar, use_container_width=True)
                    
                    with viz_tabs[2]:  # Correlation Charts
                        cols3 = st.columns(2)
                        with cols3[0]:
                            st.plotly_chart(fig_scatter, use_container_width=True)
                        with cols3[1]:
                            st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                    with viz_tabs[3]:  # Advanced Analysis
                        cols4 = st.columns(2)
                        with cols4[0]:
                            st.plotly_chart(fig_funnel, use_container_width=True)
                        with cols4[1]:
                            # Display data table with sorting and filtering
                            st.dataframe(
                                df.style.background_gradient(subset=['visual_shelf_share'], cmap='YlOrRd'),
                                use_container_width=True
                            )
                    
                except Exception as e:
                    error_msg = f'Erro ao criar visualizações: {str(e)}'
                    logger.error(error_msg)
                    st.error(error_msg)
                    
        except Exception as e:
            error_msg = f'Erro ao processar a imagem: {str(e)}'
            logger.error(error_msg)
            st.error(error_msg)

def main():
    st.title('Análise de Visual Shelf Share')
    
    # Initialize Vertex AI
    initialize_vertex_ai(PROJECT_ID, LOCATION)
    model = GenerativeModel("gemini-1.5-flash-002")

    # Load prompt
    prompt = load_prompt()

    # Create tabs including the new Quadrants and Audit Log tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Visualização", "Recomendações", "JSON Raw", "Quadrantes", "Audit Log"])
    
    # Setup the audit log in its tab
    with tab5:
        st.subheader('Audit Log')
        if 'logs' not in st.session_state:
            st.session_state.logs = []
        log_container = st.empty()
    
    # Create and add the Streamlit log handler only if not already added
    if 'logger_initialized' not in st.session_state:
        logger.handlers = []
        st_handler = StreamlitLogHandler(log_container)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        st_handler.setFormatter(formatter)
        logger.addHandler(st_handler)
        st.session_state.logger_initialized = True
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configurações")
        
        # Image selection
        st.subheader("Seleção de Imagem")
        use_default = st.checkbox("Usar imagem de exemplo", value=True)
        
        # Grid size selection
        grid_size = st.slider("Tamanho da grade", min_value=2, max_value=4, value=2, 
                            help="Número de divisões da imagem em cada dimensão")
        
        if use_default:
            try:
                with open("images/shelf.jpg", "rb") as f:
                    image_data = f.read()
                st.success("Imagem de exemplo carregada")
                st.image("images/shelf.jpg", caption="Imagem de exemplo", use_column_width=True)
            except Exception as e:
                st.error("Erro ao carregar imagem de exemplo")
                logger.error(f"Error loading default image: {str(e)}")
                image_data = None
        else:
            uploaded_file = st.file_uploader(
                "Upload de imagem da prateleira",
                type=['jpg', 'jpeg', 'png'],
                help="Selecione uma imagem de prateleira para análise"
            )
            if uploaded_file:
                image_data = uploaded_file.read()
                st.success("Imagem carregada com sucesso")
                st.image(uploaded_file, caption="Imagem carregada", use_column_width=True)
            else:
                image_data = None
        
        # Analysis button
        if st.button('Analisar Prateleira', use_container_width=True):
            if image_data:
                analyze_shelf(image_data, model, prompt, tab1, tab2, tab3, tab4, grid_size)
            else:
                st.warning("Por favor, selecione uma imagem para análise")

if __name__ == '__main__':
    main() 