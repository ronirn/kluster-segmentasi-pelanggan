import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import cophenet
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Aplikasi Pengelompokan Segmentasi Pelanggan dengan Hierarchical Clustering dan KMeans")



with st.expander("Tentang Aplikasi"):
    st.markdown("""
    ### Tujuan Aplikasi:
    Aplikasi ini dirancang untuk melakukan **segmentasi pelanggan** menggunakan algoritma **Hierarchical Clustering** dan **KMeans** berdasarkan atribut-atribut pelanggan, seperti jenis kelamin, status perkawinan, usia, pendidikan, pendapatan, pekerjaan, dan ukuran pemukiman.

    Dengan aplikasi ini, pengguna dapat:
    - Memilih fitur yang relevan untuk digunakan dalam analisis clustering.
    - Melakukan **preprocessing data** yang mencakup standarisasi data numerik dan encoding untuk fitur kategorikal.
    - Menggunakan **Hierarchical Clustering** dan **KMeans** untuk clustering data berdasarkan parameter yang dipilih.
    - Mendapatkan hasil clustering yang tersegmentasi berdasarkan kluster yang terbentuk.
    - Mengunduh hasil clustering dalam format CSV untuk analisis lebih lanjut.
    """)

data = pd.read_csv('Clustering.csv')

st.subheader("Eksplorasi Data")

tab1, tab2, tab3 = st.tabs(["Deskripsi Data", "Statistik Deskriptif", "Visualisasi Korelasi"])

with tab1:
    st.write("Dataset yang digunakan:")
    st.dataframe(data.head())

with tab2:
    st.write("Statistik Deskriptif:")
    st.write(data.describe())

with tab3:
    st.write("Heatmap Korelasi:")
    corr_matrix = data.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

st.subheader("Pilih fitur untuk clustering:")
features = st.multiselect("Fitur yang digunakan:", data.columns.tolist(), default=data.columns.tolist())

if features:
    st.subheader("Preprocessing Data")
    selected_data = data[features]
    st.write("Data yang dipilih:", selected_data.head())
    
    le = LabelEncoder()
    for col in selected_data.select_dtypes(include='object').columns:
        selected_data[col] = le.fit_transform(selected_data[col])
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(selected_data)
    st.write("Data setelah standarisasi:")
    st.write(pd.DataFrame(scaled_data, columns=features).head())

    # Sidebar for Algorithm Selection and Sliders
    st.sidebar.subheader("Pilih Algoritma Clustering")
    algorithm = st.sidebar.selectbox("Pilih algoritma clustering:", ["Hierarchical", "KMeans"], index=0)

    if algorithm == "Hierarchical":
        st.subheader("Hierarchical Clustering")
        st.write("Penjelasan Metode:")
        with st.expander("Penjelasan tentang setiap metode"):
            st.markdown("""
        **Metode Ward** adalah metode linkage yang mengoptimalkan jumlah kuadrat perbedaan dalam mengelompokkan data. 
        Metode ini menggabungkan dua kluster dengan cara yang meminimalkan peningkatan total variansi dalam kluster yang terbentuk.
        Ward biasanya menghasilkan kluster yang lebih seimbang dan cocok untuk data dengan banyak fitur.
        
        **Metode Complete (atau Maximum Linkage)** mengukur jarak antar dua kluster berdasarkan jarak terjauh antara dua titik dalam kluster yang berbeda.
        Metode ini mengutamakan kluster yang lebih homogen dan lebih sensitif terhadap outlier karena mempertimbangkan jarak maksimum.
        
        **Metode Average Linkage** mengukur jarak antar dua kluster berdasarkan rata-rata jarak antara semua pasangan titik dalam dua kluster yang berbeda.
        Metode ini lebih seimbang dibandingkan dengan metode lengkap dan lebih toleran terhadap outlier.
        
        **Metode Single (atau Nearest Point Linkage)** mengukur jarak antar dua kluster berdasarkan jarak terdekat antara dua titik dalam kluster yang berbeda.
        Metode ini seringkali menghasilkan kluster yang lebih memanjang dan sensitif terhadap noise atau outlier.
        """)

        method = st.sidebar.selectbox("Pilih metode linkage:", ["ward", "complete", "average", "single"], index=0)
        linkage_matrix = linkage(scaled_data, method=method)
        
        st.write("Dendrogram:")
        fig, ax = plt.subplots(figsize=(10, 6))
        dendrogram(linkage_matrix, labels=data.index, truncate_mode="level", p=5)
        plt.title("Dendrogram")
        plt.xlabel("Data Points")
        plt.ylabel("Distance")
        st.pyplot(fig)
        

        st.sidebar.subheader("Pengaturan Kluster")
# Tentukan jumlah kluster berdasarkan slider
        num_clusters = st.sidebar.slider("Pilih jumlah kluster:", min_value=2, max_value=10, value=3, step=1)
        cluster_labels = fcluster(linkage_matrix, num_clusters, criterion="maxclust")
        data["Cluster"] = cluster_labels

# Evaluasi performa Hierarchical Clustering
        st.subheader("Evaluasi Performa Hierarchical Clustering")
        silhouette_avg = silhouette_score(scaled_data, data["Cluster"])
        dbi = davies_bouldin_score(scaled_data, data["Cluster"])
        chi = calinski_harabasz_score(scaled_data, data["Cluster"])

        st.write(f"- **Silhouette Score**: {silhouette_avg:.2f} (Semakin mendekati 1, semakin baik kluster yang terbentuk)")
        st.write(f"- **Davies-Bouldin Index**: {dbi:.2f} (Semakin kecil, semakin baik)")
        st.write(f"- **Calinski-Harabasz Index**: {chi:.2f} (Semakin besar, semakin baik)")

        
        # Visualisasi PCA untuk Hierarchical Clustering
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(scaled_data)
        data["PCA1"] = reduced_data[:, 0]
        data["PCA2"] = reduced_data[:, 1]

        st.write("Visualisasi Kluster:")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", palette="tab10", data=data, ax=ax)
        plt.title("Visualisasi Hasil Hierarchical Clustering dengan PCA")
        st.pyplot(fig)

if algorithm == "KMeans":
    st.subheader("KMeans Clustering")
    
        # Tambahkan Elbow Method
    st.write("### Metode Elbow")
    st.write("Metode Elbow digunakan untuk menentukan jumlah kluster optimal dengan menghitung inertia (jumlah kuadrat jarak dalam kluster).")
    max_clusters = st.slider("Jumlah maksimum kluster untuk Elbow Method:", 2, 15, 10)
    inertia_values = []

    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        inertia_values.append(kmeans.inertia_)
        
        # Visualisasi Elbow Method
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.plot(range(1, max_clusters + 1), inertia_values, marker='o', linestyle='--', color='b')
    plt.title("Elbow Method")
    plt.xlabel("Jumlah Kluster")
    plt.ylabel("Inertia")
    plt.grid()
    st.pyplot(fig)

    st.write("Dari grafik di atas, Anda dapat memilih jumlah kluster yang optimal pada titik 'elbow', yaitu ketika penurunan inertia mulai melambat.")
    kmeans_clusters = st.sidebar.slider("Jumlah kluster:", 2, 10, 3)
    kmeans = KMeans(n_clusters=kmeans_clusters, random_state=42)
    data["Cluster"] = kmeans.fit_predict(scaled_data)
    

    st.write("Hasil KMeans Clustering:")
    st.dataframe(data)

    # Evaluasi KMeans Clustering
    st.subheader("Evaluasi Performa KMeans")
    silhouette_avg = silhouette_score(scaled_data, data["Cluster"])
    dbi = davies_bouldin_score(scaled_data, data["Cluster"])
    chi = calinski_harabasz_score(scaled_data, data["Cluster"])

    st.write(f"- **Silhouette Score**: {silhouette_avg:.2f} (Semakin mendekati 1, semakin baik kluster yang terbentuk)")
    st.write(f"- **Davies-Bouldin Index**: {dbi:.2f} (Semakin kecil, semakin baik)")
    st.write(f"- **Calinski-Harabasz Index**: {chi:.2f} (Semakin besar, semakin baik)")

    # Tambahkan diagram visual jika diperlukan
    st.write("Visualisasi dengan PCA:")
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(scaled_data)
    data["PCA1"] = reduced_data[:, 0]
    data["PCA2"] = reduced_data[:, 1]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", palette="viridis", data=data, ax=ax)
    plt.title("Visualisasi Hasil Clustering dengan PCA")
    st.pyplot(fig)



st.subheader("Diagram Metrik Kluster")

cluster_counts = data["Cluster"].value_counts()
total_customers = len(data)

# Membuat plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=data, x="Cluster", palette="viridis", ax=ax)

# Menambahkan label dan persentase dengan penyesuaian tampilan
for p in ax.patches:
    height = p.get_height()
    percentage = (height / total_customers) * 100
    ax.annotate(f'{height} ({percentage:.2f}%)',
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', fontsize=9, color='black', 
                xytext=(0, 5), textcoords='offset points')  # Menambahkan spasi agar label tidak bertumpuk

# Menambahkan judul dan label
plt.title("Distribusi Pelanggan berdasarkan Kluster", fontsize=16)
plt.xlabel("Kluster", fontsize=14)
plt.ylabel("Jumlah Pelanggan", fontsize=14)

# Mengatur tampilan grid untuk kemudahan membaca
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Menampilkan grafik
st.pyplot(fig)

st.subheader("Metrik Kluster")

cluster_counts = data["Cluster"].value_counts()
tabs = [f"Kluster {cluster_num}" for cluster_num in cluster_counts.index]
tab_objs = st.tabs(tabs)

for idx, cluster_num in enumerate(cluster_counts.index):
    with tab_objs[idx]:
        count = cluster_counts[cluster_num]
        percentage = (count / len(data)) * 100
        st.markdown(f"### Kluster {cluster_num}")
        st.metric(
            label="Jumlah Pelanggan", 
            value=f"{count}", 
            delta=f"{percentage:.2f}% dari total pelanggan", 
            delta_color="normal"
        )

st.subheader("Unduh Hasil")
csv = data.to_csv(index=False).encode('utf-8')

# Mengubah warna dan gaya tombol secara terbatas
st.download_button(
    label="Download Hasil Clustering",
    data=csv,
    file_name="data_hasil_klustering.csv",
    mime="text/csv",
    use_container_width=True
)

# Watermark Teks
st.markdown(
    """
    <div style="position: fixed; bottom: 10px; right: 15px; font-size: 14px; color: rgba(0, 0, 0, 0.5);">
        Dibuat oleh Roni
    </div>
    """, 
    unsafe_allow_html=True
)


