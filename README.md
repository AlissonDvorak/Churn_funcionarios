# Projeto de Previsão de Rotatividade de Funcionários

Este projeto utiliza modelos de machine learning para prever a rotatividade de funcionários com base em características específicas. Siga os passos abaixo para configurar e executar o projeto.

## Pré-requisitos

Certifique-se de ter o [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) e [Python](https://www.python.org/downloads/) instalados no seu sistema.

## Passos para Executar o Projeto

### 1. Criar o Ambiente Conda

Crie um novo ambiente Conda para o projeto e ative-o:

```sh
conda create --name meu_ambiente python=3.11
conda activate meu_ambiente
```
### 2. Instalar Dependências
```sh
pip install -r requirements.txt
```
### 3. Executar o Aplicativo

```sh
cd app
python main.py
```

O aplicativo estará disponível em http://localhost:8000.

### 3. Treinar um Novo Modelo (Opcional)

Se desejar treinar um novo modelo, você pode usar o endpoint de treinamento. Defina o parâmetro melhorModelo como true para melhorar o modelo existente. Faça uma solicitação GET para o endpoint / com os seguintes parâmetros:


```sh
http://localhost:8000/?vizualizar_distribuicao_classes=false&melhorModelo=true&gerar_baseline=false
```

### **atencao esse passo pode ser bastante demorado dependendo de sua maquina, ele rodara o PyCaret para avaliar o modelo que ira se sair melhor para seu dataset.**



###5. Utilizar o Endpoint de Previsão

Para prever se um funcionário sairá ou não, use o endpoint /realizar_previsao. Envie uma solicitação POST com os parâmetros do funcionário no corpo da solicitação. Exemplo de payload:

```sh
[
    {
        "satisfaction_level": 0.1,
        "last_evaluation": 0.76,
        "number_project": 588,
        "average_montly_hours": 190,
        "time_spend_company": 12,
        "Work_accident": 1,
        "promotion_last_5years": 0,
        "Departments": "sales",
        "salary": "low"
    },
    {
        "satisfaction_level": 0.65,
        "last_evaluation": 0.70,
        "number_project": 3,
        "average_montly_hours": 130,
        "time_spend_company": 2,
        "Work_accident": 1,
        "promotion_last_5years": 0,
        "Departments": "technical",
        "salary": "medium"
    }
]

```

Contribuição
Sinta-se à vontade para contribuir com o projeto. Por favor, envie um pull request ou abra uma issue para discutir melhorias.

Licença
Este projeto está licenciado sob a MIT License.
