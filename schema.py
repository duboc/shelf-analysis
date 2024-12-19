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

SHELF_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "analise_prateleira": {
            "type": "object",
            "properties": {
                "fabricantes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "nome": {
                                "type": "string",
                                "description": "Nome do fabricante"
                            },
                            "presente": {
                                "type": "boolean",
                                "description": "Indica se o fabricante está presente na prateleira (Sim/Não)"
                            },
                            "contagem_remedios": {
                                "type": "integer",
                                "description": "Número de remédios do fabricante visíveis na prateleira"
                            },
                            "visual_shelf_share": {
                                "type": "number",
                                "format": "float",
                                "description": "Percentual de Visual Shelf Share do fabricante"
                            }
                        },
                        "required": ["nome", "presente", "contagem_remedios", "visual_shelf_share"]
                    }
                },
                "observacoes": {
                    "type": "string",
                    "description": "Observações sobre a metodologia utilizada na análise"
                },
                "consideracoes": {
                    "type": "string",
                    "description": "Considerações gerais sobre a distribuição dos fabricantes na prateleira"
                }
            },
            "required": ["fabricantes", "observacoes", "consideracoes"]
        }
    },
    "required": ["analise_prateleira"]
} 