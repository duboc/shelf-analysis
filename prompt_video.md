# Análise de Vídeo de Prateleira de Farmácia

Analise detalhadamente o vídeo de uma prateleira de farmácia que estou te fornecendo. O objetivo é entender a dinâmica e interações na prateleira ao longo do vídeo. Para isso, foque nos seguintes aspectos:

1.  **Identificação de Produtos e Fabricantes**:
    *   Liste os produtos e seus fabricantes que são claramente identificáveis no vídeo.
    *   Note se algum produto é pego, manuseado ou colocado na prateleira.

2.  **Interações com a Prateleira**:
    *   Descreva quaisquer interações humanas com a prateleira (e.g., clientes olhando produtos, funcionários reabastecendo).
    *   Se houver interações, descreva quais produtos/marcas foram o foco da interação.

3.  **Visibilidade de Produtos**:
    *   Há produtos que se destacam mais ou que ficam visíveis por mais tempo?
    *   Alguma área da prateleira recebe mais atenção ou movimento?

4.  **Observações Gerais do Vídeo**:
    *   Forneça um resumo geral dos eventos e observações importantes capturados no vídeo relacionados à prateleira.
    *   Existem insights sobre o comportamento do consumidor ou a gestão da prateleira que podem ser inferidos do vídeo?

5.  **Formato da Resposta**:
    *   Apresente os resultados em um formato JSON estruturado.
    *   O JSON deve ter chaves principais como `analise_video_prateleira` contendo seções para `produtos_identificados`, `interacoes_observadas`, `visibilidade_destacada`, e `resumo_geral`.

Exemplo da estrutura JSON esperada:
{
  "analise_video_prateleira": {
    "produtos_identificados": [
      {
        "nome_produto": "Produto Exemplo A",
        "fabricante": "Fabricante X",
        "interacao_observada": "Cliente pegou o produto aos 0:35"
      }
    ],
    "interacoes_observadas": [
      {
        "timestamp_inicio": "0:32",
        "timestamp_fim": "0:45",
        "descricao": "Cliente observando a seção de analgésicos, pegou o Produto Exemplo A.",
        "produtos_foco": ["Produto Exemplo A"]
      }
    ],
    "visibilidade_destacada": {
      "produtos_mais_visiveis": ["Produto Exemplo B", "Produto Exemplo C"],
      "areas_maior_atencao": ["Prateleira superior, seção central"]
    },
    "resumo_geral": "O vídeo mostra um cliente interagindo com a seção de analgésicos por aproximadamente 15 segundos. Houve também um breve reabastecimento de produtos na prateleira inferior pelo funcionário aos 1:10."
  }
}

Por favor, retorne APENAS o objeto JSON, sem texto adicional antes ou depois.
