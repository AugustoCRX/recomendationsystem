import pandas as pd
import numpy as np
import streamlit as st
import time 

from sklearn.metrics.pairwise import cosine_similarity

#Leitura dos dados
df = pd.read_csv(r"https://raw.githubusercontent.com/AugustoCRX/recomendationsystem/refs/heads/main/DADOS/spotify_songs.csv").dropna().drop_duplicates(subset="track_name").reset_index(drop=True)
reorder_list = df.columns.tolist()
reorder_list.remove("track_name")
reorder_list.insert(0,"track_name")
df = df[reorder_list]
prep_df = df.copy()



#Classe do sistema de recomendação
class RecomendationSystem():
    def __init__(self, sample):
        self.sample = sample
    #Método responsável por calcular a similaridade de cosseno
    def calculate_cosine(self):
        #Separação dos dummies
        dummies = prep_df[prep_df.columns[22::]]
        #Seração da coluna que será utilizada como parametro
        danceability = prep_df[prep_df.columns[11]]
        #União dos dummies com a coluna parâmetro
        df_cosine_calculation = pd.concat([dummies,danceability], axis = 1)
        #Reshape da tabela para que seja possível executar a cosine similarity
        df_sample = np.array(df_cosine_calculation.iloc[self.sample]).reshape(1, -1)
        cosine_column = cosine_similarity(df_cosine_calculation, df_sample)
        #Adiciona a coluna ao dataframe
        df['Cosine_Dist'] = cosine_column
        return df
    #Método responsável por executar dijkstra na aplicação
    def dijkstra(self):
        df_cosine = self.calculate_cosine()
        distances = df['Cosine_Dist']
        distance_dict = {i:10**7 for i in range(len(distances))}
        #Armazena as distâncias dentro de uma lista
        for i in range(len(distances)):
            distance_dict[i] = 1/distances[i] #Formula para cálculo das distâncias
        #Reconstroi o dicionário com as distâncias reordenadas
        distance_dict = dict(sorted(distance_dict.items(), key = lambda item: item[1], reverse=False))
        # Reordenar DataFrame baseado nas distâncias
        df_new_order = [i for i in distance_dict]
        df_teste_sorted = df_cosine.iloc[df_new_order]
        return df_teste_sorted

st.markdown("# Find the SPOT!")

# Entrada de texto para o filtro
entrada_usuario = st.text_input("Digite o nome da música:")

# Filtrar os nomes que contenham o texto digitado
opcoes_filtradas = prep_df[prep_df['track_name'].str.contains(entrada_usuario, case=False)]

# Dropdown exibindo apenas as opções filtradas
escolha = st.selectbox("Selecione um nome:", opcoes_filtradas)

#Botão para execução do código de dijkstra
if st.button("Procure músicas parecidas!"):
    #Inicio do tempo de execução do código
    application_start = time.time()
    # Executar o código de dijkstra e buscar o índice selecionado
    recomendation = RecomendationSystem(np.where(prep_df['track_name'] == escolha))
    df_recommendation = recomendation.dijkstra()
    st.write("Músicas que fazem o seu tipo!")
    df_show = df_recommendation.head(11).loc[df["track_name"] != escolha, ["track_name","track_artist","track_album_name","playlist_name","track_id","playlist_genre"]].rename(columns=
        {"track_name":"Nome da música",
         "track_artist":"Nome do artista",
         "track_album_name": "Nome do album",
         "playlist_genre": "Genero da playlist"}
        )

    st.dataframe(df_show.drop(["playlist_name", "track_id"], axis = 1))
    #Dimensionamento da div para divisão dos nomes das músicas e do botão
    col1, col2 = st.columns([5, 1])
    with col1:
    # Loop para mostrar o nome das músicas
        for index, row in df_show.iterrows():
            nome_musica = row['Nome da música']
            
            # Exibir oo nome da música
            st.write(f"""
            **Música**: {nome_musica}
            """)
    with col2:
        for index, row in df_show.iterrows():
            # Armazena o link da música do Spotify
            music_id = row['track_id']
            spotify_link = f"https://open.spotify.com/track/{music_id}"
            
            # Botão que leva ao link do Spotify
            st.markdown(f"[🔈]({spotify_link})", unsafe_allow_html=True)
    #Finalização do tempo de execução
    st.write(f"O tempo de execução da aplicação é de {(time.time() - application_start):.2f} segundos")
