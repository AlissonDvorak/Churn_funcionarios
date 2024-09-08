import pandas as pd
from tabulate import tabulate
from ydata_profiling import ProfileReport
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import seaborn as sns
import pickle
from pycaret.classification import *
import matplotlib.pyplot as plt
import io
import base64
from fastapi.responses import StreamingResponse

# %matplotlib inline


def importarDados():
    dados = pd.read_csv("./Churn_funcionarios/archive/HR_Dataset.csv")
    dados.head()
    return dados


def transformarDados(dados):
    dados = dados.dropna(axis=0)
    return dados


def dummy(dados):
    X = dados.drop(['left'], axis=1)
    y = dados['left']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    dummy_stratified = DummyClassifier(strategy='stratified')
    dummy_stratified.fit(X_train, y_train)
    acuracia_stratified = dummy_stratified.score(X_test, y_test) * 100

    dummy_most_frequent = DummyClassifier(strategy='most_frequent')
    dummy_most_frequent.fit(X_train, y_train)
    y_pred_most_frequent = dummy_most_frequent.predict(X_test)

    report_most_frequent = classification_report(
        y_test, y_pred_most_frequent, output_dict=True)
    report_table = []
    for label, metrics in report_most_frequent.items():
        if label != 'accuracy':
            row = [label] + [metrics.get(m, '')
                             for m in ['precision', 'recall', 'f1-score', 'support']]
            report_table.append(row)

    headers = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
    formatted_report = tabulate(report_table, headers=headers, tablefmt='grid')

    return {
        'dummy_stratified': f"A baseline via dummy stratified foi de {acuracia_stratified:.2f}%",
        'dummy_most_frequent': formatted_report
    }



def salvar_distribuicao_classes(dados):
    contagem_classes = dados['left'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    contagem_classes.plot(kind='bar', ax=ax)
    plt.title('Distribuição das Classes')
    plt.xlabel('Classe')
    plt.ylabel('Número de Ocorrências')
    plt.xticks(ticks=[0, 1], labels=['Não Saída', 'Saída'], rotation=0)

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)


    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()

    return img_str


def aplicar_one_hot_encoder(dados):
    categorias = dados.select_dtypes(include=['object']).columns
    num_dados = dados.drop(columns=categorias)
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_categorias = encoder.fit_transform(dados[categorias])
    
    with open('./Churn_funcionarios/arquivos_pickle/encoder.pkl', 'wb') as file:
        pickle.dump(encoder, file)

    # Ajustar os nomes das colunas para remover espaços extras
    encoded_df = pd.DataFrame(encoded_categorias, columns=[col.replace(
        ' ', '') for col in encoder.get_feature_names_out(categorias)])

    dados_encoded = pd.concat([num_dados, encoded_df], axis=1)
    
    
    return dados_encoded, encoder


def compararModelos(dados_encoded):
    s = ClassificationExperiment()
    s.setup(dados_encoded, target='left', session_id=123)

    best_model = s.compare_models()

    print("Melhor Modelo:")
    print(best_model)

    final_model = s.finalize_model(best_model)

    print("Melhor Modelo:")
    print(best_model)
    print("Detalhes do Melhor Modelo:")
    print(s.pull())

    with open('./Churn_funcionarios/arquivos_pickle/melhor_modelo.pkl', 'wb') as file:
        pickle.dump(final_model, file)
    print("Modelo salvo em 'melhor_modelo.pkl'")

    return final_model


def realizar_pipeline(vizualizar_distribuicao_classes, gerar_baseline,  melhorModelo):
    dados = importarDados()
    dados = transformarDados(dados)
    dados_encoded, encoder = aplicar_one_hot_encoder(dados)

    if melhorModelo:
        best_model = compararModelos(dados_encoded)
        print("Modelo Final Treinado:")
        print()
        return f"Modelo Final Treinado: {best_model} e salvo em 'melhor_modelo.pkl'"

    # Outras funções que você pode querer chamar
    if vizualizar_distribuicao_classes:
        return visualizar_distribuicao_classes(dados)
    if gerar_baseline:
        return dummy(dados)


def churnRF(dados_brutos):
    modeloRF = pickle.load(open('./Churn_funcionarios/arquivos_pickle/melhor_modelo.pkl', 'rb'))

    colunas_esperadas = [
        'satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours',
        'time_spend_company', 'Work_accident', 'promotion_last_5years',
        'Departments_RandD', 'Departments_accounting', 'Departments_hr',
        'Departments_management', 'Departments_marketing', 'Departments_product_mng',
        'Departments_sales', 'Departments_support', 'Departments_technical',
        'salary_low', 'salary_medium'
    ]

    df_dados_brutos = pd.DataFrame(dados_brutos)

    categorias = ['Departments', 'salary']
    df_categorias = df_dados_brutos[categorias]
    df_dados_numericos = df_dados_brutos.drop(columns=categorias)

    encoder = pickle.load(open('./Churn_funcionarios/arquivos_pickle/encoder.pkl', 'rb'))
    encoder.fit(df_categorias)

    df_encoded = pd.DataFrame(encoder.transform(df_categorias), columns=encoder.get_feature_names_out(categorias))

    dados_encoded = pd.concat([df_dados_numericos.reset_index(drop=True), df_encoded.reset_index(drop=True)], axis=1)

    for col in colunas_esperadas:
        if col not in dados_encoded.columns:
            dados_encoded[col] = 0

    dados_encoded = dados_encoded[colunas_esperadas]

    previsao = modeloRF.predict(dados_encoded)
        
    return previsao





# def randoForest(dados_numericos):
#     X = dados_numericos.drop(['left'], axis=1)
#     y = dados_numericos['left']
#     modelo = RandomForestClassifier(class_weight={0: 1, 1: 3}, random_state=42)
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42)

#     modelo.fit(X_train, y_train)
#     y_pred = modelo.predict(X_test)

#     with open('melhor_modelorf.pkl', 'wb') as file:
#         pickle.dump(modelo, file)

#     cm = confusion_matrix(y_test, y_pred)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[
#                 'Não Saída', 'Saída'], yticklabels=['Não Saída', 'Saída'])
#     plt.title('Matriz de Confusão')
#     plt.xlabel('Classe Predita')
#     plt.ylabel('Classe Real')
#     plt.show()

#     # Curva ROC e AUC
#     y_prob = modelo.predict_proba(X_test)[:, 1]
#     fpr, tpr, _ = roc_curve(y_test, y_prob)
#     auc = roc_auc_score(y_test, y_prob)

#     plt.figure(figsize=(8, 6))
#     plt.plot(fpr, tpr, marker='o', label=f'AUC = {auc:.2f}')
#     plt.title('Curva ROC')
#     plt.xlabel('Taxa de Falsos Positivos')
#     plt.ylabel('Taxa de Verdadeiros Positivos')
#     plt.legend()
#     plt.show()

#     print("Classification Report para o conjunto de teste:")
#     print(classification_report(y_test, y_pred))

#     # Validação cruzada
#     scores = cross_val_score(modelo, X, y, cv=5)
#     print(f"Acurácia média na validação cruzada: {scores.mean():.2f}")


# vizualizar o sweetviz do dataset
# def relatorio(dados):
#     profile = ProfileReport(dados)
#     profile.to_file("sweetviz.html")