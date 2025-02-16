# **Laporan Proyek Machine Learning - Wahyu Azizi**

## **Project Overview**

Perkembangan platform game **Steam** menunjukkan pertumbuhan yang signifikan dalam beberapa tahun terakhir, baik dari segi jumlah game yang tersedia maupun fitur-fitur yang ditawarkan. Steam telah mengalami lonjakan koleksi judul game dalam lima tahun terakhir. Pada tahun 2019, Steam memiliki sekitar 8.134 judul game, yang meningkat menjadi 14.532 judul pada tahun 2023. Hal ini menunjukkan pertumbuhan yang impresif dan menjadikan Steam sebagai salah satu platform dengan koleksi game terluas di industri.[[1](https://eraspace.com/artikel/post/sejumlah-perbandingan-steam-dan-epic-games-mana-yang-terbaik)]

Banyak faktor yang mempengaruhi pertumbuhan platform Steam, salah satunya adalah **sistem rekomendasi** yang dikembangkan pada platform ini. Sistem ini memungkinkan pengguna mendapatkan rekomendasi yang dipersonalisasi. Dengan menggunakan algoritma yang menganalisis preferensi dan perilaku pengguna, platform dapat menawarkan rekomendasi game yang lebih relevan. Ini membantu pengguna menemukan game yang sesuai dengan minat mereka, sehingga meningkatkan kemungkinan mereka untuk melakukan pembelian.[[2](https://www.semanticscholar.org/paper/Implementasi-Chatbot-Telegram-untuk-Meningkatkan-Yuwan-Soelistijadi/fe61864360798ba62bada1ea70b790a0bd6e5f69)]  
Personalisasi memungkinkan interaksi yang lebih baik antara pengguna dan platform. Ketika pengguna merasa bahwa rekomendasi yang diberikan sesuai dengan keinginan mereka, mereka cenderung lebih terlibat dan loyal terhadap platform tersebut.[[3](https://www.semanticscholar.org/paper/Analisis-Penerapan-UI-UX-Dalam-Meningkatkan-Pada-Syafei-Hidayatullah/ca7cfbaa16740716159ba1660203ce012d26ba16)]  

Rekomendasi yang tepat juga dapat meningkatkan tingkat konversi penjualan. Pengguna lebih cenderung membeli game yang direkomendasikan jika mereka merasa bahwa permainan tersebut sesuai dengan preferensi mereka. Dengan demikian, rekomendasi yang relevan dapat meningkatkan pendapatan bagi platform.[[4](https://www.semanticscholar.org/paper/Implementasi-Chatbot-Telegram-untuk-Meningkatkan-Yuwan-Soelistijadi/fe61864360798ba62bada1ea70b790a0bd6e5f69)] Selain itu, pengalaman yang dipersonalisasi membuat pengguna merasa dihargai dan diperhatikan. Hal ini dapat membangun loyalitas jangka panjang terhadap platform, karena pengguna cenderung kembali untuk mencari rekomendasi di masa depan.  
[[5](https://www.semanticscholar.org/paper/Analisis-Penerapan-UI-UX-Dalam-Meningkatkan-Pada-Syafei-Hidayatullah/ca7cfbaa16740716159ba1660203ce012d26ba16)]  

Untuk membangun sistem rekomendasi, terdapat beberapa teknik yang umum digunakan, antara lain:  

1. **Collaborative Filtering (CF):**  
   Pendekatan ini berfokus pada perilaku pengguna dan interaksi antar pengguna untuk memberikan rekomendasi.  

2. **Content-Based Filtering (CBF):**  
   Teknik ini berfokus pada karakteristik item untuk memberikan rekomendasi. Metode ini menganalisis fitur-fitur dari item dan membandingkannya dengan preferensi pengguna. Misalnya, dalam sistem rekomendasi film, CBF mempertimbangkan genre, aktor, dan sutradara dari film yang telah ditonton oleh pengguna sebelumnya.  

Dalam proyek ini, saya akan menggunakan data rekomendasi game pada Steam yang tersedia di Kaggle. Data tersebut akan menjadi dasar dalam membangun sistem rekomendasi berbasis machine learning.  


## **Business Understanding**

### **Problem Statements**

1. Bagaimana cara mengembangkan sistem rekomendasi yang dipersonalisasi untuk platform Steam menggunakan data pengguna?
2. Bagaimana tingkat kepuasan pengguna terhadap game yang dimainkan?
3. Game apa yang populer dimainkan oleh pengguna?

### **Goals**

1. Membuat sistem rekomendasi yang dipersonalisasi untuk platform Steam menggunakan pendekatan Collaborative Filtering dan Content-based Filtering berbasis model deep learning.
2. Mengevaluasi tingkat kepuasan pengguna terhadap game yang dimainkan berdasarkan rasio ulasan positif.
3. Mengidentifikasi game yang paling populer berdasarkan jumlah ulasan, rating, harga dan durasi bermain.

### **Solutions**

#### **1. Pengembangan Sistem Rekomendasi**
- **Pendekatan:** 
  - **Collaborative Filtering:** Rekomendasi berdasarkan pola interaksi pengguna dan game.
  - **Content-Based Filtering:** Rekomendasi berdasarkan atribut game, seperti genre dan deskripsi.
- **Implementasi:** 
  - Gunakan **deep learning** untuk Collaborative Filtering.
  - Evaluasi model dengan metrik seperti RMSE dan Precision.

#### **2. Eksplorasi Data (EDA)**
- **Analisis Data:**
  - Visualisasikan distribusi data seperti harga game, ulasan, dan durasi bermain.
  - Identifikasi faktor yang memengaruhi harga, ulasan, dan durasi bermain.
- **Tools:**
  - Gunakan **Python** (Pandas, Matplotlib, Seaborn) untuk manipulasi data dan visualisasi.
  - Tampilkan hasil dengan grafik sederhana seperti histogram atau heatmap. 

**Hasil yang Diharapkan:**
- Sistem rekomendasi yang relevan untuk pengguna.
- Wawasan faktor-faktor penting yang memengaruhi aspek game di Steam.

## **Data Understanding**

Lakukan proses data understanding, ini termasuk data loading, memeriksa tipe data pada variabel, missing value, dan duplikasi data, setelah itu memberikan detail deskripsi variabel

### **Data Loading**  

Data yang digunakan merupakan data game pada sebuah platform game yaitu Steam. data bisa didapatkan dengan mengklik [link ini](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam)


```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("antonkozyriev/game-recommendations-on-steam")

print("Path to dataset files:", path)
```

    Path to dataset files: /kaggle/input/game-recommendations-on-steam


    Reading games_metadata.json ...
    Reading users.csv ...
    Reading games.csv ...
    Reading recommendations.csv ...
    Selesai!


Di projek ini saya akan menggunakan games_metadata.json, games.csv, dan recommendations.csv

### **Identifikasi Missing Value, Data Duplikasi, dan Tipe Data**

Selanjutnya adalah memeriksa beberapa hal dalam data, seperti missing value, data duplikasi, dan memerika tipe data pada variabel.

#### **df_games**
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>app_id</th>
      <th>title</th>
      <th>date_release</th>
      <th>win</th>
      <th>mac</th>
      <th>linux</th>
      <th>rating</th>
      <th>positive_ratio</th>
      <th>user_reviews</th>
      <th>price_final</th>
      <th>price_original</th>
      <th>discount</th>
      <th>steam_deck</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13500</td>
      <td>Prince of Persia: Warrior Within™</td>
      <td>2008-11-21</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>Very Positive</td>
      <td>84</td>
      <td>2199</td>
      <td>9.99</td>
      <td>9.99</td>
      <td>0.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22364</td>
      <td>BRINK: Agents of Change</td>
      <td>2011-08-03</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>Positive</td>
      <td>85</td>
      <td>21</td>
      <td>2.99</td>
      <td>2.99</td>
      <td>0.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>113020</td>
      <td>Monaco: What's Yours Is Mine</td>
      <td>2013-04-24</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>Very Positive</td>
      <td>92</td>
      <td>3722</td>
      <td>14.99</td>
      <td>14.99</td>
      <td>0.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>226560</td>
      <td>Escape Dead Island</td>
      <td>2014-11-18</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>Mixed</td>
      <td>61</td>
      <td>873</td>
      <td>14.99</td>
      <td>14.99</td>
      <td>0.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>249050</td>
      <td>Dungeon of the ENDLESS™</td>
      <td>2014-10-27</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>Very Positive</td>
      <td>88</td>
      <td>8784</td>
      <td>11.99</td>
      <td>11.99</td>
      <td>0.0</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>


    Shape: (50872, 13)
    
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 50872 entries, 0 to 50871
    Data columns (total 13 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   app_id          50872 non-null  int64  
     1   title           50872 non-null  object 
     2   date_release    50872 non-null  object 
     3   win             50872 non-null  bool   
     4   mac             50872 non-null  bool   
     5   linux           50872 non-null  bool   
     6   rating          50872 non-null  object 
     7   positive_ratio  50872 non-null  int64  
     8   user_reviews    50872 non-null  int64  
     9   price_final     50872 non-null  float64
     10  price_original  50872 non-null  float64
     11  discount        50872 non-null  float64
     12  steam_deck      50872 non-null  bool   
    dtypes: bool(4), float64(3), int64(3), object(3)
    memory usage: 3.7+ MB



    Jumlah Data Duplikasi: 0
    






<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Daftar Missing Value:</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>app_id</th>
      <td>0</td>
    </tr>
    <tr>
      <th>title</th>
      <td>0</td>
    </tr>
    <tr>
      <th>date_release</th>
      <td>0</td>
    </tr>
    <tr>
      <th>win</th>
      <td>0</td>
    </tr>
    <tr>
      <th>mac</th>
      <td>0</td>
    </tr>
    <tr>
      <th>linux</th>
      <td>0</td>
    </tr>
    <tr>
      <th>rating</th>
      <td>0</td>
    </tr>
    <tr>
      <th>positive_ratio</th>
      <td>0</td>
    </tr>
    <tr>
      <th>user_reviews</th>
      <td>0</td>
    </tr>
    <tr>
      <th>price_final</th>
      <td>0</td>
    </tr>
    <tr>
      <th>price_original</th>
      <td>0</td>
    </tr>
    <tr>
      <th>discount</th>
      <td>0</td>
    </tr>
    <tr>
      <th>steam_deck</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Data ```df_games``` memiliki jumlah row sebanyak 50872 row data, dengan total variabel sebanyak 13 variabel. Beberapa tipe data yang ada adalah float, integer, boolean, dan juga object (termasuk string, dan date), tidak ada masalah dalam tipe data pada data ```df_games```. Selain itu tidak ditemukan data yang hilang ataupun terduplikasi.

#### **df_recom**




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>app_id</th>
      <th>helpful</th>
      <th>funny</th>
      <th>date</th>
      <th>is_recommended</th>
      <th>hours</th>
      <th>user_id</th>
      <th>review_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>975370</td>
      <td>0</td>
      <td>0</td>
      <td>2022-12-12</td>
      <td>True</td>
      <td>36.3</td>
      <td>51580</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>304390</td>
      <td>4</td>
      <td>0</td>
      <td>2017-02-17</td>
      <td>False</td>
      <td>11.5</td>
      <td>2586</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1085660</td>
      <td>2</td>
      <td>0</td>
      <td>2019-11-17</td>
      <td>True</td>
      <td>336.5</td>
      <td>253880</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>703080</td>
      <td>0</td>
      <td>0</td>
      <td>2022-09-23</td>
      <td>True</td>
      <td>27.4</td>
      <td>259432</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>526870</td>
      <td>0</td>
      <td>0</td>
      <td>2021-01-10</td>
      <td>True</td>
      <td>7.9</td>
      <td>23869</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



    Shape: (41154794, 8)
    
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 41154794 entries, 0 to 41154793
    Data columns (total 8 columns):
     #   Column          Dtype  
    ---  ------          -----  
     0   app_id          int64  
     1   helpful         int64  
     2   funny           int64  
     3   date            object 
     4   is_recommended  bool   
     5   hours           float64
     6   user_id         int64  
     7   review_id       int64  
    dtypes: bool(1), float64(1), int64(5), object(1)
    memory usage: 2.2+ GB



    Jumlah Data Duplikasi: 0
    





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Daftar Missing Value:</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>app_id</th>
      <td>0</td>
    </tr>
    <tr>
      <th>helpful</th>
      <td>0</td>
    </tr>
    <tr>
      <th>funny</th>
      <td>0</td>
    </tr>
    <tr>
      <th>date</th>
      <td>0</td>
    </tr>
    <tr>
      <th>is_recommended</th>
      <td>0</td>
    </tr>
    <tr>
      <th>hours</th>
      <td>0</td>
    </tr>
    <tr>
      <th>user_id</th>
      <td>0</td>
    </tr>
    <tr>
      <th>review_id</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```df_recom``` merupakan data rekomendasi pada game, memiliki total row sebanyak 41154794 row, dan memiliki 8 variabel. Tidak ada masalah pada tipe data pada variabel, dan juga tidak ditemukan missing value dan data yang terduplikasi.

#### **df_meta**



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>app_id</th>
      <th>description</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13500</td>
      <td>Enter the dark underworld of Prince of Persia ...</td>
      <td>[Action, Adventure, Parkour, Third Person, Gre...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22364</td>
      <td></td>
      <td>[Action]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>113020</td>
      <td>Monaco: What's Yours Is Mine is a single playe...</td>
      <td>[Co-op, Stealth, Indie, Heist, Local Co-Op, St...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>226560</td>
      <td>Escape Dead Island is a Survival-Mystery adven...</td>
      <td>[Zombies, Adventure, Survival, Action, Third P...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>249050</td>
      <td>Dungeon of the Endless is a Rogue-Like Dungeon...</td>
      <td>[Roguelike, Strategy, Tower Defense, Pixel Gra...</td>
    </tr>
  </tbody>
</table>
</div>



    Shape: (50872, 3)
    
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 50872 entries, 0 to 50871
    Data columns (total 3 columns):
     #   Column       Non-Null Count  Dtype 
    ---  ------       --------------  ----- 
     0   app_id       50872 non-null  int64 
     1   description  50872 non-null  object
     2   tags         50872 non-null  object
    dtypes: int64(1), object(2)
    memory usage: 1.2+ MB


Sebelum mengidentifikasi data duplikasi, perlu diperhatikan bahwa value pada variabel ```tags``` adalah dalam bentuk list, untuk itu variabel ini harus diubah menjadi string





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>app_id</th>
      <th>description</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>37510</th>
      <td>1867270</td>
      <td>If you like platform and adventure games, get ...</td>
      <td>Action Adventure Action-Adventure Arcade Platf...</td>
    </tr>
    <tr>
      <th>2221</th>
      <td>587470</td>
      <td>Chroma Lab is a VR particle physics sandbox. S...</td>
      <td>Simulation Indie Casual VR Physics Sandbox</td>
    </tr>
    <tr>
      <th>7925</th>
      <td>1496320</td>
      <td>An exciting adventure through space and time a...</td>
      <td>Casual Adventure Hidden Object Puzzle Investig...</td>
    </tr>
    <tr>
      <th>26392</th>
      <td>617630</td>
      <td>This is a puzzle game that focuses on the adve...</td>
      <td>Casual Strategy Adventure Indie Puzzle Beautiful</td>
    </tr>
    <tr>
      <th>36565</th>
      <td>776890</td>
      <td>In this game you will have an exciting immersi...</td>
      <td>Simulation Indie Casual Clicker</td>
    </tr>
  </tbody>
</table>
</div>




    Jumlah Data Duplikasi: 0
    



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Daftar Missing Value:</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>app_id</th>
      <td>0</td>
    </tr>
    <tr>
      <th>description</th>
      <td>0</td>
    </tr>
    <tr>
      <th>tags</th>
      <td>1244</td>
    </tr>
  </tbody>
</table>
</div>



data pada ```df_meta``` sendiri memiliki total row sebanyak 50872 entries, dengan jumlah variabel sebanyak 3 variabel. Tidak ada masalah pada tipe data, data duplikasi juga nol, tapi terdapat nilai pada variabel tags yang hilang, sebanyak 1244 tags.

___

Menggabungkan beberapa data menjadi satu sebelum melakukan analysis dan data preparation, serta identifikasi missing value.



cek missing value pada df_games_rec:
```
    app_id                0
    title                 0
    date_release          0
    win                   0
    mac                   0
    linux                 0
    rating                0
    positive_ratio        0
    user_reviews          0
    price_final           0
    price_original        0
    discount              0
    steam_deck            0
    helpful           13262
    funny             13262
    date              13262
    is_recommended    13262
    hours             13262
    user_id           13262
    review_id         13262
    dtype: int64

```


cek missing value pada df_games_meta:
```
    app_id               0
    title                0
    date_release         0
    win                  0
    mac                  0
    linux                0
    rating               0
    positive_ratio       0
    user_reviews         0
    price_final          0
    price_original       0
    discount             0
    steam_deck           0
    description          0
    tags              1244
    dtype: int64
```


### **Exploratory Data Analysis - Deskripsi Variabel**

Berikut daftar deksripsi variabel yang ada pada data yang sudah di load.

#### df_games
| **Nama Variabel**   | **Deskripsi**                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `app_id`            | ID unik untuk setiap game.                                                 |
| `title`             | Judul dari game.                                                          |
| `date_release`      | Tanggal rilis game dalam format `YYYY-MM-DD`.                             |
| `win`               | Apakah game tersedia untuk platform Windows (`True/False`).               |
| `mac`               | Apakah game tersedia untuk platform MacOS (`True/False`).                |
| `linux`             | Apakah game tersedia untuk platform Linux (`True/False`).                |
| `rating`            | Peringkat keseluruhan dari game berdasarkan ulasan pengguna (e.g., Very Positive). |
| `positive_ratio`    | Persentase ulasan positif dari total ulasan.                              |
| `user_reviews`      | Jumlah total ulasan pengguna untuk game tersebut.                         |
| `price_final`       | Harga akhir setelah diskon (jika ada).                                    |
| `price_original`    | Harga asli sebelum diskon.                                                |
| `discount`          | Besarnya diskon dalam bentuk persentase.                                  |
| `steam_deck`        | Apakah game kompatibel dengan Steam Deck (`True/False`).                  |

---

#### df_recommendation
| **Nama Variabel**   | **Deskripsi**                                                        |
|----------------------|----------------------------------------------------------------------|
| `app_id`            | ID unik untuk game yang diulas.                                     |
| `helpful`           | Jumlah vote "membantu" pada ulasan tersebut.                        |
| `funny`             | Jumlah vote "lucu" pada ulasan tersebut.                            |
| `date`              | Tanggal ulasan dibuat dalam format `YYYY-MM-DD`.                    |
| `is_recommended`    | Apakah pengguna merekomendasikan game tersebut (`True/False`).      |
| `hours`             | Jumlah waktu yang dihabiskan untuk memainkan game tersebut (dalam jam). |
| `user_id`           | ID unik dari pengguna yang memberikan ulasan.                      |
| `review_id`         | ID unik untuk ulasan tertentu.                                      |

---

#### df_metadata
| **Nama Variabel**   | **Deskripsi**                                                               |
|----------------------|---------------------------------------------------------------------------|
| `app_id`            | ID unik untuk setiap game.                                               |
| `description`       | Deskripsi singkat tentang game.                                          |
| `tags`              | Kategori atau tag yang terkait dengan game, seperti genre dan fitur.     |


### **Exploratory Data Analysis - Univariate Analysis**  

Berikut distribusi untuk kategorical features, pada data bisa diketahui bahwa 98% lebih game yang terdapat pada platform Steam mendukung system operasi Windows, sedang sebanyak 75% game tersedia untuk system operasi Mac, dan 82,2% game tersedia dalam system operasi Linux, dan yang terakhir adalah hampir semua game mendukung steam deck.
Distribusi untuk kategori rating dapat disimpulkan bahwa kebanyakan game pada Steam dengan rating positive, artinya Steam sangat memperhatikan game yang masuk kedalam platform mereka.


    
![png](gambar_files/gambar_46_0.png)
    


Pada numerical features, yang pertama distribusi dari variable positive_ratio sangat menunjukkan hal positive karna banyaknya rating yang positive sehingga rasio positive yang tinggi juga sangat banyak. Untuk variable price original dan price final, tidak menunjukkan perbedaan yang signifikan, dan ternyata sangat banyak game dengan Harga yang relative rendah bahkan gratis.



    
![png](gambar_files/gambar_48_0.png)
    


### **Exploratory Data Analysis - Multivariate Analysis**

Selanjutnya, menganalisis game popular berdasarkan beberapa kategori yang berhubungan dengan melakukan analisis ultivariat. Pertama Game terpopuler berdasarkan kategori rating terhadap user_review dimana rating ```Very Positive``` sangat dominan dengan jumlah lebih dari 7e6 user review, beberapa game dalam kategori ini yaitu ```Counter-Strike: Global Offensive``` dengan urutan teratas, diikuti oleh game ```PUBG, dan Terraria``` sebagai top tiga game teratas dari kategori ini.
Selanjutnya berdasarkan rating terhadap price_final, aplikasi teratas adalah ```Clickteam Fusion 2.5 Developer Upgrade``` dengan harga 300 dollar, diikuti oleh software ```Aart Curvy 3D 3.0 dan Houdini Indie```, akan tetapi aplikasi ini mendapat rating yang positive.
Dan terakhir adalah game teratas berdasarkan total jam bermain, dimana ```The Quarry, Bad Rats: the Rat's Revenge, Clicker Heroes 2``` menjadi top tiga game dengan jumlah hours tertinggi dan dengan rating yang sangat positive.



    
![png](gambar_files/gambar_51_0.png)
    



    
![png](gambar_files/gambar_51_1.png)
    



    
![png](gambar_files/gambar_51_2.png)
    


## **Data Preparation & Model Development dengan Content-Based Filtering**

### **Preparation for Content-Based Filtering**

Waktunya mempersiapkan data untuk mengembangkan model machine learning agar bisa digunakan dengan maksimal. Sebelum itu, dikarenakan terbatasnya memori yang bisa digunakan untuk memproses data yang sangat banyak ini, terlebih dahulu dilakukannya filtering data dengan mengambil beberapa sample data, disini menggunakan data mulai dari tahun 2020, kemudian hanyak mengambil sample 10000 dari data tersebut, sehingga yang awalnya data dengan total rows 25880823 menjadi 36700 lalu mengambil 10000 dari data tersebut. Selanjutnya memilih variabel-variabel yang dibutuhkan dalam content-based filtering ini, diantaranya yaitu ```app_id, title, win, mac, Linux, rating, tags```.


    Number of rows after date filtering: 25880823




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>app_id</th>
      <th>helpful</th>
      <th>funny</th>
      <th>date</th>
      <th>is_recommended</th>
      <th>hours</th>
      <th>user_id</th>
      <th>review_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>975370</td>
      <td>0</td>
      <td>0</td>
      <td>2022-12-12</td>
      <td>True</td>
      <td>36.3</td>
      <td>51580</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>703080</td>
      <td>0</td>
      <td>0</td>
      <td>2022-09-23</td>
      <td>True</td>
      <td>27.4</td>
      <td>259432</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>526870</td>
      <td>0</td>
      <td>0</td>
      <td>2021-01-10</td>
      <td>True</td>
      <td>7.9</td>
      <td>23869</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>306130</td>
      <td>0</td>
      <td>0</td>
      <td>2021-10-10</td>
      <td>True</td>
      <td>8.6</td>
      <td>45425</td>
      <td>5</td>
    </tr>
    <tr>
      <th>7</th>
      <td>730</td>
      <td>0</td>
      <td>0</td>
      <td>2021-11-30</td>
      <td>False</td>
      <td>157.5</td>
      <td>63209</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>



### **TF-IDF Vectorizer**

Selanjutnya membangun sistem rekomendasi sederhana berdasarkan tags yang ada pada games dengan pendekatan TF-IDF. TF-IDF (Term Frequency - Inverse Document Frequency) adalah metode yang digunakan dalam pemrosesan bahasa alami (NLP) untuk mengubah teks menjadi representasi numerik berdasarkan pentingnya kata dalam sebuah dokumen relatif terhadap koleksi dokumen (corpus).TF-IDF digunakan dalam berbagai aplikasi seperti information retrieval, text mining, dan machine learning untuk NLP.  

**1. Apa Itu TF-IDF?**
**TF-IDF (Term Frequency - Inverse Document Frequency)** adalah metode yang digunakan dalam pemrosesan bahasa alami (**NLP**) untuk mengubah teks menjadi representasi numerik berdasarkan pentingnya kata dalam sebuah dokumen relatif terhadap koleksi dokumen (*corpus*).  

TF-IDF digunakan dalam berbagai aplikasi seperti **information retrieval**, **text mining**, dan **machine learning untuk NLP**.

---

**2. Rumus TF-IDF**
**a) Term Frequency (TF)**
Menunjukkan seberapa sering kata muncul dalam sebuah dokumen.  

$$
TF(t, d) = \frac{f(t, d)}{N}
$$

- \( f(t, d) \) = jumlah kemunculan kata \( t \) dalam dokumen \( d \)
- \( N \) = total jumlah kata dalam dokumen \( d \)

**b) Inverse Document Frequency (IDF)**
Mengukur seberapa penting kata dalam keseluruhan koleksi dokumen.  

$$
IDF(t) = \log{\frac{D}{1 + DF(t)}}
$$

- \( D \) = total jumlah dokumen dalam corpus
- \( DF(t) \) = jumlah dokumen yang mengandung kata \( t \)
- Ditambahkan **+1** dalam penyebut untuk menghindari pembagian dengan nol.

**c) TF-IDF Score**
Dihitung dengan mengalikan nilai TF dan IDF:

$$
TF\text{-}IDF(t, d) = TF(t, d) \times IDF(t)
$$

Semakin tinggi nilai **TF-IDF**, semakin penting kata tersebut dalam dokumen.

dari hasi yang ditunjukkan pada table, tags family dimiliki oleh game King of Retail dan Tamarin, selain itu Tamarin memiliki tag stylized, controller dan metroidvania. ini hanya beberapa contoh ayng dapat ditampilkan.


```python
tfidf = TfidfVectorizer()

tfidf.fit(df_content_based['tags'])
tfidf.get_feature_names_out()
```




    array(['1980s', '1990', '2d', '360', '3d', '40k', '4x', '5d', '6dof',
           'abstract', 'access', 'action', 'addictive', 'adventure',
           'agriculture', 'aliens', 'alternate', 'ambient', 'america',
           'american', 'and', 'animation', 'anime', 'apocalyptic', 'arcade',
           'archery', 'arena', 'artificial', 'arts', 'assassin', 'asymmetric',
           'asynchronous', 'atmospheric', 'attack', 'atv', 'audio', 'auto',
           'automation', 'automobile', 'awkward', 'base', 'baseball', 'based',
           'basketball', 'battle', 'battler', 'beat', 'beautiful',
           'benchmark', 'bikes', 'bit', 'blood', 'bmx', 'board', 'book',
           'boss', 'bowling', 'boxing', 'builder', 'building', 'bullet',
           'campaign', 'capitalism', 'card', 'cartoon', 'cartoony', 'casual',
           'cats', 'character', 'chess', 'choices', 'choose', 'cinematic',
           'city', 'class', 'classic', 'click', 'clicker', 'co', 'cold',
           'collectathon', 'collector', 'colony', 'colorful', 'combat',
           'comedy', 'comic', 'competitive', 'conspiracy', 'content',
           'control', 'controller', 'controls', 'conversation', 'cooking',
           'cozy', 'craft', 'crafting', 'crawler', 'creature', 'cricket',
           'crime', 'crowdfunded', 'crpg', 'cult', 'customization', 'cute',
           'cyberpunk', 'cycling', 'dark', 'dating', 'death', 'deckbuilder',
           'deckbuilding', 'deduction', 'defense', 'demons', 'design',
           'destruction', 'detective', 'development', 'difficult',
           'dinosaurs', 'diplomacy', 'documentary', 'dog', 'down', 'dragons',
           'drama', 'drawn', 'driving', 'dungeon', 'dungeons', 'dynamic',
           'dystopian', 'early', 'economy', 'editing', 'editor', 'education',
           'electronic', 'em', 'emotional', 'endings', 'epic', 'episodic',
           'escape', 'esports', 'events', 'experience', 'experimental',
           'exploration', 'faith', 'family', 'fantasy', 'farming', 'fast',
           'feature', 'female', 'fi', 'fiction', 'fighter', 'fighting',
           'film', 'first', 'fishing', 'flight', 'fmv', 'football', 'foreign',
           'fps', 'free', 'friendly', 'funny', 'futuristic', 'gambling',
           'game', 'gamemaker', 'games', 'gaming', 'generation', 'god',
           'golf', 'gore', 'gothic', 'grand', 'graphics', 'great', 'grid',
           'gun', 'hack', 'hacking', 'hand', 'hardware', 'heist', 'hell',
           'hentai', 'hero', 'hex', 'hidden', 'historical', 'history',
           'hobby', 'hockey', 'horror', 'horses', 'humor', 'hunting', 'idler',
           'ii', 'illuminati', 'illustration', 'immersive', 'indie',
           'instrumental', 'intelligence', 'intentionally', 'interactive',
           'inventory', 'investigation', 'isometric', 'jet', 'job', 'jrpg',
           'jump', 'keeper', 'kickstarter', 'lego', 'lemmings', 'level',
           'lgbtq', 'life', 'like', 'linear', 'local', 'logic', 'loot',
           'looter', 'lore', 'lovecraftian', 'machine', 'magic', 'mahjong',
           'management', 'manipulation', 'mars', 'martial', 'massively',
           'match', 'matter', 'mature', 'mechs', 'medical', 'medieval',
           'memes', 'metroidvania', 'military', 'mini', 'minigames',
           'minimalist', 'mining', 'mmorpg', 'moba', 'mod', 'moddable',
           'modeling', 'modern', 'motocross', 'motorbike', 'mouse',
           'movement', 'movie', 'multiplayer', 'multiple', 'music', 'musou',
           'mystery', 'mythology', 'narration', 'narrative', 'nature',
           'naval', 'ninja', 'noir', 'nonlinear', 'nostalgia', 'novel',
           'nsfw', 'nudity', 'object', 'offroad', 'old', 'on', 'online',
           'only', 'op', 'open', 'otome', 'outbreak', 'own', 'paced',
           'parkour', 'parody', 'party', 'pause', 'perma', 'person',
           'philosophical', 'photo', 'physics', 'pinball', 'pirates', 'pixel',
           'platformer', 'play', 'player', 'point', 'political', 'politics',
           'pool', 'post', 'precision', 'procedural', 'production',
           'programming', 'protagonist', 'psychedelic', 'psychological',
           'publishing', 'puzzle', 'pve', 'pvp', 'quick', 'racing', 'rails',
           'real', 'realistic', 'reboot', 'relaxing', 'remake', 'replay',
           'resource', 'retro', 'rhythm', 'rich', 'robots', 'rock',
           'roguelike', 'roguelite', 'roguevania', 'romance', 'rome', 'room',
           'royale', 'rpg', 'rpgmaker', 'rts', 'rugby', 'runner', 'rush',
           'sailing', 'sandbox', 'satire', 'scare', 'school', 'sci',
           'science', 'score', 'screen', 'scroller', 'sequel', 'sexual',
           'shoot', 'shooter', 'shop', 'short', 'side', 'silent', 'sim',
           'simulation', 'simulator', 'singleplayer', 'skateboarding',
           'skating', 'skiing', 'slash', 'sniper', 'snooker', 'snow',
           'snowboarding', 'soccer', 'social', 'software', 'sokoban',
           'solitaire', 'souls', 'soundtrack', 'space', 'spaceships',
           'spectacle', 'spelling', 'split', 'sports', 'stealth', 'steam',
           'steampunk', 'stick', 'story', 'strategy', 'stylized', 'submarine',
           'superhero', 'supernatural', 'surreal', 'survival', 'swordplay',
           'tabletop', 'tactical', 'tactics', 'tanks', 'team', 'tennis',
           'text', 'third', 'thriller', 'time', 'to', 'top', 'touch', 'tower',
           'trackir', 'trading', 'traditional', 'training', 'trains',
           'transhumanism', 'transportation', 'travel', 'trivia', 'turn',
           'tutorial', 'twin', 'typing', 'underground', 'underwater',
           'unforgiving', 'up', 'utilities', 'value', 'vampire', 'vehicular',
           'video', 'vikings', 'villain', 'violent', 'vision', 'visual',
           'voice', 'volleyball', 'voxel', 'vr', 'walking', 'war', 'wargame',
           'warhammer', 'web', 'well', 'werewolves', 'western', 'wholesome',
           'with', 'word', 'workshop', 'world', 'wrestling', 'written',
           'your', 'zombies'], dtype=object)




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>addictive</th>
      <th>satire</th>
      <th>economy</th>
      <th>story</th>
      <th>illustration</th>
      <th>loot</th>
      <th>documentary</th>
      <th>dynamic</th>
      <th>top</th>
      <th>design</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Fling to the Finish</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Fast &amp; Furious: Spy Racers Rise of SH1FT3R</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Cargo Commander</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Timore 5</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Love Breakout</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Intralism</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Planet of the Apes: Last Frontier</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.329216</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>World of Contraptions</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.208868</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Survive the Nights</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Thea 2: The Shattering</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.105343</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



### **Cosine Similarity**

**Cosine Similarity** adalah metode untuk mengukur kesamaan antara dua vektor berdasarkan sudut (*angle*) di antara keduanya dalam ruang berdimensi tinggi.  

Metode ini sering digunakan dalam **text mining**, **information retrieval**, dan **recommender systems** untuk membandingkan dokumen atau item.

**Rumus Cosine Similarity**
Diberikan dua vektor **A** dan **B**, kesamaan kosinus dihitung dengan:

$$
\cos{\theta} = \frac{A \cdot B}{\|A\| \times \|B\|}
$$

Di mana:
- \( A \cdot B \) = **dot product** antara vektor \( A \) dan \( B \)
- \( \|A\| \) = **norma** (panjang) dari vektor \( A \), dihitung sebagai:

$$
\|A\| = \sqrt{\sum_{i=1}^{n} A_i^2}
$$

- \( \|B\| \) = norma dari vektor \( B \)
- \( \theta \) = sudut antara dua vektor

Nilai **Cosine Similarity** berada dalam rentang **\([-1,1]\)**:
- **1** → Vektor identik (sangat mirip)
- **0** → Tidak ada hubungan (ortogonal)
- **-1** → Vektor berlawanan arah (sangat berbeda)





Observasi dari Sampel:

- Once Again memiliki nilai tinggi dengan Cris Tales (0.318) dan Duck Life (0.178), menunjukkan adanya kesamaan fitur antar game tersebut.
- Sunrider: Liberation Day - Captain's Edition memiliki kesamaan tertinggi dengan Trap Legend (0.498), menunjukkan bahwa tags mereka mungkin sangat mirip.
- Cooking Festival dan Duck Life memiliki kesamaan relatif tinggi (0.329), mungkin karena tema kasual atau mekanisme gameplay yang mirip.
- Beberapa game seperti Reanimation Inc. dan ZANGEKI WARP memiliki nilai yang sangat rendah (mendekati 0), menunjukkan bahwa game tersebut mungkin sangat berbeda dari lainnya.


```python
cos_sim_df = pd.DataFrame(
    cos_sim, 
    index=df_content_based['title'], 
    columns=df_content_based['title']
)
print(f"Shape: {cos_sim_df.shape}")
cos_sim_df.sample(5, axis=1).sample(10, axis=0)
```

    Shape: (10000, 10000)



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>title</th>
      <th>Slipways</th>
      <th>Grandma's Footsteps</th>
      <th>Battle Chess: Game of Kings™</th>
      <th>Shift Happens</th>
      <th>Cat or Bread?</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Touch Down Football Solitaire</th>
      <td>0.046189</td>
      <td>0.099313</td>
      <td>0.145846</td>
      <td>0.042350</td>
      <td>0.250667</td>
    </tr>
    <tr>
      <th>Ultimate MMA</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Lost Flame</th>
      <td>0.123116</td>
      <td>0.042398</td>
      <td>0.000000</td>
      <td>0.029326</td>
      <td>0.016206</td>
    </tr>
    <tr>
      <th>I Am Overburdened</th>
      <td>0.193142</td>
      <td>0.097416</td>
      <td>0.018863</td>
      <td>0.094862</td>
      <td>0.065430</td>
    </tr>
    <tr>
      <th>Just Cause</th>
      <td>0.044209</td>
      <td>0.070049</td>
      <td>0.039673</td>
      <td>0.118838</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>AI: Art Impostor</th>
      <td>0.000000</td>
      <td>0.033093</td>
      <td>0.318630</td>
      <td>0.059000</td>
      <td>0.023834</td>
    </tr>
    <tr>
      <th>Super Night Riders</th>
      <td>0.011551</td>
      <td>0.094288</td>
      <td>0.000000</td>
      <td>0.039926</td>
      <td>0.013000</td>
    </tr>
    <tr>
      <th>Warp Frontier</th>
      <td>0.132807</td>
      <td>0.191244</td>
      <td>0.000000</td>
      <td>0.024089</td>
      <td>0.011953</td>
    </tr>
    <tr>
      <th>Eville</th>
      <td>0.041998</td>
      <td>0.078427</td>
      <td>0.118639</td>
      <td>0.318035</td>
      <td>0.028296</td>
    </tr>
    <tr>
      <th>NiGHT SIGNAL</th>
      <td>0.005009</td>
      <td>0.380246</td>
      <td>0.000000</td>
      <td>0.004593</td>
      <td>0.275138</td>
    </tr>
  </tbody>
</table>
</div>



### **Mendapatkan Rekomendasi**



hasil rekomendasi dari system yang dibangun, dengan mencoba menginput data game dengan title BlackJack Math, diperoleh hasil rekomendasi dengan rata-rata similarity scorenya sama dengan 0.65, beberapa game yang direkomendasikan yaitu ```Journey of Greed, Love Letter, Splendor, Avalon Legends Solitaire, Pathfinder Avdentures```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>app_id</th>
      <th>title</th>
      <th>date_release</th>
      <th>win</th>
      <th>mac</th>
      <th>linux</th>
      <th>rating</th>
      <th>positive_ratio</th>
      <th>user_reviews</th>
      <th>price_final</th>
      <th>price_original</th>
      <th>discount</th>
      <th>steam_deck</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2580</th>
      <td>1341220</td>
      <td>BlackJack Math</td>
      <td>2020-08-20</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>Very Positive</td>
      <td>88</td>
      <td>60</td>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.0</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>similarity score</th>
      <th>app_id</th>
      <th>date_release</th>
      <th>win</th>
      <th>mac</th>
      <th>linux</th>
      <th>rating</th>
      <th>positive_ratio</th>
      <th>user_reviews</th>
      <th>price_final</th>
      <th>price_original</th>
      <th>discount</th>
      <th>steam_deck</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Journey of Greed</td>
      <td>0.666917</td>
      <td>1032790</td>
      <td>2019-10-23</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>Very Positive</td>
      <td>90</td>
      <td>1127</td>
      <td>6.99</td>
      <td>6.99</td>
      <td>0.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Love Letter</td>
      <td>0.652366</td>
      <td>926520</td>
      <td>2018-10-24</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>Very Positive</td>
      <td>89</td>
      <td>174</td>
      <td>6.99</td>
      <td>6.99</td>
      <td>0.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Splendor</td>
      <td>0.652366</td>
      <td>376680</td>
      <td>2015-09-17</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>Very Positive</td>
      <td>83</td>
      <td>660</td>
      <td>3.99</td>
      <td>9.99</td>
      <td>60.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Avalon Legends Solitaire 2</td>
      <td>0.611270</td>
      <td>512260</td>
      <td>2016-10-04</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>Very Positive</td>
      <td>88</td>
      <td>50</td>
      <td>9.99</td>
      <td>9.99</td>
      <td>0.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pathfinder Adventures</td>
      <td>0.632800</td>
      <td>480640</td>
      <td>2017-06-15</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>Mixed</td>
      <td>67</td>
      <td>562</td>
      <td>9.99</td>
      <td>9.99</td>
      <td>0.0</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



## **Data Preparation & Model Development dengan Collaborative Filtering**

### **Data Preparation for Collaborative Filtering**  

Selanjutnya membangun sistem rekomendasi dengan pendekatan Collaborative Filtering. Pertama melakukan filtering data untuk mengambil sample dengan rentang tertentu agar cukup memori untuk memproses semua ini. Gunakan fungsi yang sudah dibuat sebelumnya, kemudian merge dengan data games untuk mengambil feature yang akan diproses nantinya.




    Number of rows after date filtering: 25880823




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>app_id</th>
      <th>helpful</th>
      <th>funny</th>
      <th>date</th>
      <th>is_recommended</th>
      <th>hours</th>
      <th>user_id</th>
      <th>review_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>975370</td>
      <td>0</td>
      <td>0</td>
      <td>2022-12-12</td>
      <td>True</td>
      <td>36.3</td>
      <td>51580</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>703080</td>
      <td>0</td>
      <td>0</td>
      <td>2022-09-23</td>
      <td>True</td>
      <td>27.4</td>
      <td>259432</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>526870</td>
      <td>0</td>
      <td>0</td>
      <td>2021-01-10</td>
      <td>True</td>
      <td>7.9</td>
      <td>23869</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>306130</td>
      <td>0</td>
      <td>0</td>
      <td>2021-10-10</td>
      <td>True</td>
      <td>8.6</td>
      <td>45425</td>
      <td>5</td>
    </tr>
    <tr>
      <th>7</th>
      <td>730</td>
      <td>0</td>
      <td>0</td>
      <td>2021-11-30</td>
      <td>False</td>
      <td>157.5</td>
      <td>63209</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>



Jangan lupa untuk memilih variabel yang sesuai untuk diproses selanjutnya. Pada Collaborative Filtering ini akan menggunakan fitur ```app_id, user_id, hours, is_recommended, dan rating```


    user_id             int64
    app_id              int64
    hours             float64
    is_recommended       bool
    rating             object
    dtype: object


Setelah itu lakukan proses encode pada beberapa variabel yang diperlukan, seperti rating, app_id, dan user_id. Karna variabel rating merupakan kategorikal dengan total 9 jenis rating, dari Overwhelmingly Negative hingga Overwhelmingly Positive. dari kesembilan rating tersebut akan disandi menjadi numerik dari 1-9, agar bisa diproses oleh model nanti.





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>app_id</th>
      <th>hours</th>
      <th>is_recommended</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>18566048</th>
      <td>3417439</td>
      <td>1735950</td>
      <td>2.0</td>
      <td>True</td>
      <td>7</td>
    </tr>
    <tr>
      <th>18599285</th>
      <td>10690950</td>
      <td>415950</td>
      <td>0.1</td>
      <td>False</td>
      <td>4</td>
    </tr>
    <tr>
      <th>6914590</th>
      <td>5474975</td>
      <td>825630</td>
      <td>56.0</td>
      <td>True</td>
      <td>8</td>
    </tr>
    <tr>
      <th>18734833</th>
      <td>6748574</td>
      <td>890930</td>
      <td>0.1</td>
      <td>True</td>
      <td>4</td>
    </tr>
    <tr>
      <th>19208492</th>
      <td>11548326</td>
      <td>768450</td>
      <td>5.1</td>
      <td>True</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>



Selanjutnya melakukan proses encoding user_id dan app_id, kemudian melakukan pemetaan keduanya kedalam dataframe. Selain itu juga memproses rating menjadi float untuk proses selanjutnya dalam proses data normalization



### **Data Normalization**

Proses transformasi data dalam kode ini terdiri dari dua langkah utama. Pertama, kolom hours dimodifikasi menggunakan metode .apply() dengan conditional weighting. Jika pengguna merekomendasikan suatu item (is_recommended == True), maka nilai hours dikalikan 1.25, memberikan bobot lebih tinggi karena dianggap lebih berharga. Sebaliknya, jika pengguna tidak merekomendasikan, nilai hours dikalikan 0.75, mengurangi pengaruhnya dalam model rekomendasi. Transformasi ini bertujuan untuk menyesuaikan kontribusi waktu bermain terhadap sistem rekomendasi, sehingga jam bermain pengguna yang puas memiliki dampak lebih besar dibandingkan yang tidak puas.

Langkah kedua adalah normalisasi rating menggunakan Min-Max Scaling, di mana nilai rating dikonversi ke skala 0 hingga 1 menggunakan rumus:

$$
X_{\text{normalized}} = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}
$$

di mana:
- \( X \) adalah nilai asli yang akan dinormalisasi  
- \( X_{\text{min}} \) adalah nilai minimum dalam dataset  
- \( X_{\text{max}} \) adalah nilai maksimum dalam dataset  
- \( X_{\text{normalized}} \) adalah nilai hasil normalisasi dalam rentang **[0, 1]**  

Normalisasi ini membantu algoritma **Machine Learning dan Collaborative Filtering** agar bekerja lebih optimal dengan data yang memiliki skala seragam.

Setiap rating dikurangi dengan rating minimum dalam dataset (min_rating), kemudian dibagi dengan selisih antara rating maksimum (max_rating) dan minimum. Teknik ini digunakan untuk memastikan bahwa perbedaan skala rating antar pengguna tidak mempengaruhi sistem rekomendasi secara berlebihan, serta meningkatkan kinerja model dalam perhitungan kesamaan (similarity metrics).

Kedua proses ini merupakan bagian dari data preprocessing dalam Collaborative Filtering, yang bertujuan untuk meningkatkan kualitas rekomendasi dengan memastikan bahwa data memiliki bobot yang sesuai dan skala yang seragam. 



### **Membagi Data untuk Training dan Validasi**

Lakukan split data untuk kedua model yang akan dibangun menggunakan Fungsi ```train_test_split``` dari pustaka ```scikit-learn``` untuk membagi dataset menjadi dua bagian: ```training set``` dan ```validation set```. Ini dilakukan agar model dapat belajar dari training set dan diuji performanya di validation set sebelum digunakan pada data baru.


```python
# Train test split untuk model 1
X_train_h, X_val_h, y_train_h, y_val_h = train_test_split(
    x,
    y_1, 
    test_size=0.2, 
    random_state=42
)

# Train test split untuk model 2
X_train_r, X_val_r, y_train_r, y_val_r = train_test_split(
    x,
    y_2, 
    test_size=0.2, 
    random_state=42
)
```

### **Bangun Model**

**1. RecommenderNet**  

Kelas RecommenderNet merupakan model rekomendasi berbasis Neural Collaborative Filtering dengan embedding layers untuk pengguna dan aplikasi.

a. Inisialisasi Model (__init__ function)

- num_users → Jumlah total pengguna dalam dataset.
- num_apps → Jumlah total aplikasi dalam dataset.
- embedding_size → Ukuran vektor embedding yang digunakan untuk merepresentasikan pengguna dan aplikasi dalam ruang laten.

Di dalam fungsi inisialisasi, terdapat empat embedding layers:

- self.user_embedding → Layer embedding untuk pengguna dengan inisialisasi He Normal dan regularisasi L2 untuk mencegah overfitting.
- self.user_bias → Layer embedding tambahan untuk bias pengguna.
- self.app_embedding → Layer embedding untuk aplikasi dengan metode yang sama seperti pengguna.
- self.app_bias → Layer embedding tambahan untuk bias aplikasi.

b. Forward Pass (call function)
- Model menerima input dalam bentuk pasangan (user_id, app_id)
- Vektor embedding untuk pengguna dan aplikasi diekstrak dari embedding layers
- Dot product antara vektor pengguna dan vektor aplikasi dihitung untuk mendapatkan skor kecocokan.
- Bias pengguna dan aplikasi ditambahkan ke hasil perhitungan.
- Aktivasi sigmoid diterapkan agar skor berada dalam rentang 0 hingga 1, yang merepresentasikan probabilitas rekomendasi. [[6](https://keras.io/examples/structured_data/collaborative_filtering_movielens/)]

**2. CustomEarlyStopping (Callback untuk Stop Training Otomatis)**  
Kelas CustomEarlyStopping digunakan untuk menghentikan pelatihan lebih awal jika loss tidak mengalami peningkatan setelah sejumlah epoch tertentu (patience).

Mekanisme:
- Jika loss membaik, model akan menyimpan nilai best_loss dan reset counter (self.wait = 0).
- Jika loss tidak membaik dalam sejumlah epoch (default: 5), pelatihan dihentikan secara otomatis.

**3. Learning Rate Scheduler (scheduler function)**  
Fungsi ini mengatur penurunan learning rate secara eksponensial setelah epoch ke-10.
- Epoch < 10 → Learning rate tetap.
- Epoch ≥ 10 → Learning rate dikurangi dengan faktor eksponensial e^(-0.1).

Callback lr_sc kemudian digunakan untuk mengatur learning rate selama pelatihan dengan keras.callbacks.LearningRateScheduler(scheduler).





### **Training**

Fungsi ```com_model(model)``` digunakan untuk melakukan kompilasi model sebelum proses pelatihan. Dalam fungsi ini, model dikompilasi menggunakan Binary Crossentropy sebagai fungsi loss karena tugas rekomendasi ini melibatkan klasifikasi biner, yaitu menentukan apakah suatu aplikasi cocok untuk seorang pengguna atau tidak. Optimizer yang digunakan adalah Adam dengan learning rate sebesar 0.001, yang memberikan keseimbangan antara konvergensi yang cepat dan kestabilan selama training. Selain itu, metrik evaluasi yang digunakan adalah ```Root Mean Squared Error (RMSE)``` untuk mengukur sejauh mana prediksi model berbeda dari nilai sebenarnya.

Selanjutnya, ```fungsi run_model(model, X_train, X_val, y_train, y_val)``` digunakan untuk melatih model dengan data pelatihan ```(X_train, y_train)``` serta melakukan validasi menggunakan data validasi ```(X_val, y_val)```. Model dilatih dengan batch size sebesar 16 dan berjalan selama 150 epoch, dengan parameter verbose=1 yang memastikan output training tetap ditampilkan. Fungsi ini juga menggunakan callback berupa ```CustomEarlyStopping(patience=5)```, yang menghentikan pelatihan secara otomatis jika dalam 5 epoch terakhir tidak ada perbaikan pada loss, ```serta lr_sc``` yang bertugas menyesuaikan learning rate sesuai dengan skema yang telah ditentukan.

Pada akhirnya, model RecommenderNet diinisialisasi dengan num_user, num_app, dan ukuran embedding sebesar 50 dimensi. Ukuran embedding ini berperan dalam merepresentasikan pengguna dan aplikasi dalam ruang laten, sehingga memungkinkan model untuk mempelajari pola interaksi yang lebih kompleks antara pengguna dan aplikasi yang direkomendasikan.



compile dan kemudian train model dengan memanggil fungsi yang sudah dibuat dengan memasukkan parameter yang sesuai, disini dibuat dua pendekatan, class pertama adalah hour dan yang kedua adalah rating.


```python
model_h = com_model(model)
history_h = run_model(model_h, X_train_h, X_val_h, y_train_h, y_val_h)
```

    Training dihentikan pada epoch 44 karena loss tidak membaik.
    [1m1835/1835[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 2ms/step - loss: 0.0959 - root_mean_squared_error: 0.0716 - val_loss: 0.6099 - val_root_mean_squared_error: 0.4448 - learning_rate: 3.3373e-05



```python
model_r = com_model(model)
history_r = run_model(model_r, X_train_r, X_val_r, y_train_r, y_val_r)
```

    Training dihentikan pada epoch 42 karena loss tidak membaik.
    [1m1835/1835[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 2ms/step - loss: 0.5909 - root_mean_squared_error: 0.1256 - val_loss: 0.7143 - val_root_mean_squared_error: 0.2725 - learning_rate: 4.0762e-05


### **Mendapatkan Rekomendasi Game**

Kode ini berfungsi untuk **merekomendasikan game** kepada pengguna berdasarkan model rekomendasi yang telah dilatih. Berikut penjelasan dari masing-masing bagian:

**1. Pemilihan Sampel Pengguna dan Game yang Telah Dimainkan**
Kode pertama mengambil satu pengguna secara acak dari dataset `df_collab_based` menggunakan `sample(1).iloc[0]`. Kemudian, **game yang telah dimainkan oleh pengguna tersebut** disimpan dalam `games_played_by_user`.

**2. Menentukan Game yang Belum Dimainkan oleh Pengguna**
Menggunakan operator **bitwise NOT (~)** dan metode `.isin()`, kode ini mencari **game yang belum pernah dimainkan** oleh pengguna dengan membandingkan `games_df['app_id']` dengan game yang sudah dimainkan (`games_played_by_user.app_id.values`).

- Game yang belum dimainkan kemudian **disaring ulang** agar hanya mencakup game yang terdapat dalam dictionary `app_to_app_encoded`, yang berisi encoding dari ID game.
- Semua **game yang belum dimainkan** dikonversi ke bentuk **encoded ID** agar dapat digunakan dalam model.

**3. Membuat Array Input untuk Prediksi Model**
Setelah mendapatkan daftar **game yang belum dimainkan**, kode ini membuat array `user_app_array`, yang berisi pasangan **user_id (dalam bentuk encoded) dan game_id**.

- Untuk setiap game yang belum dimainkan, pasangan **(user_id, app_id)** dibuat menggunakan **numpy hstack**, sehingga dapat digunakan sebagai input dalam model.

**4. Fungsi `recom_ids(model)` untuk Memprediksi Rekomendasi Game**
Fungsi ini menggunakan model yang telah dilatih untuk memberikan prediksi rating terhadap game yang belum dimainkan pengguna.

- Model melakukan prediksi dengan `model.predict(user_app_array).flatten()`, yang menghasilkan daftar skor rekomendasi.
- **Top 10 game** dengan skor tertinggi dipilih menggunakan **argsort()** dan dikonversi kembali ke **app_id asli** menggunakan `app_encoded_to_app`.

**5. Menampilkan Game dengan Rating Tertinggi yang Telah Dimainkan Pengguna**
Kode ini menampilkan **5 game dengan rating tertinggi** yang telah dimainkan pengguna.

- Menggunakan **sorting berdasarkan rating** (`sort_values(by='rating', ascending=False)`) dan mengambil **5 teratas**.
- Kemudian, kode mencocokkan game tersebut dengan dataset `df_games` untuk mendapatkan informasi tambahan seperti **judul game dan positive_ratio (rasio ulasan positif)**.

**6. Fungsi `top_recom_game(recom_ids, title)` untuk Menampilkan Rekomendasi Game**
Fungsi ini menampilkan **10 game rekomendasi teratas** berdasarkan model yang digunakan.

- Menggunakan dataset `df_games` untuk mendapatkan informasi judul game dan rasio ulasan positif.
- Hasil rekomendasi dicetak dengan format yang lebih rapi.

**7. Menampilkan Rekomendasi untuk Model Berbasis Waktu (`Hour-based`) dan Berbasis Rating (`Rating-based`)**
Terakhir, kode menjalankan fungsi `recom_ids()` untuk dua model berbeda:

- **Model berbasis waktu (`model_h`)**, yang menggunakan jumlah jam bermain sebagai salah satu faktor rekomendasi.
- **Model berbasis rating (`model_r`)**, yang mempertimbangkan rating dari pengguna lain sebagai faktor utama.
- Hasil dari kedua model ditampilkan dengan `top_recom_game()`, sehingga pengguna dapat melihat perbandingan antara kedua metode rekomendasi.


```
    Showing recommendations for users: 13676817
    ===========================
    Games with high ratings from user
    ---------------------------
    20XX : 92
    [1m1147/1147[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step
    ------------------------------------
    Top 10 game recommendations from Model Hour-based
    --------------------------------
    Creeper World 3: Arc Eternal : 96
    Civilization IV: Beyond the Sword : 96
    Nuclear Throne : 96
    Maitetsu:Last Run!! : 98
    Left 4 Dead 2 : 97
    RimWorld : 98
    YOU and ME and HER: A Love Story : 96
    Divinity: Original Sin 2 - Definitive Edition : 95
    Farmer Against Potatoes Idle : 96
    The Binding of Isaac: Rebirth : 97
    [1m1147/1147[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step
    ------------------------------------
    Top 10 game recommendations from Model Rating-based
    --------------------------------
    Creeper World 3: Arc Eternal : 96
    Civilization IV: Beyond the Sword : 96
    Nuclear Throne : 96
    Maitetsu:Last Run!! : 98
    Left 4 Dead 2 : 97
    RimWorld : 98
    YOU and ME and HER: A Love Story : 96
    Divinity: Original Sin 2 - Definitive Edition : 95
    Farmer Against Potatoes Idle : 96
    The Binding of Isaac: Rebirth : 97
```

User 13676817 memiliki rating tinggi untuk game 20XX (92). Berdasarkan model collaborative filtering, dua pendekatan digunakan untuk merekomendasikan game:

1. Model Hour-based dan Model Rating-based memberikan hasil rekomendasi yang identik.
2. Top 10 game yang direkomendasikan termasuk RimWorld, Left 4 Dead 2, The Binding of Isaac: Rebirth, dan Divinity: Original Sin 2, yang memiliki rating tinggi (95-98).
3. Rekomendasi ini didasarkan pada kesamaan pola permainan dan rating pengguna lain yang memiliki preferensi serupa.

## **Evaluation**

### **Evaluasi Metrik RMSE**
# Evaluasi Metrik RMSE

**Root Mean Squared Error (RMSE)** adalah metrik evaluasi yang digunakan untuk mengukur seberapa jauh prediksi model dari nilai aktual.  
RMSE merupakan akar kuadrat dari Mean Squared Error (MSE), yang berarti RMSE memiliki satuan yang sama dengan variabel target, sehingga lebih mudah untuk diinterpretasikan.

## 📌 Formula RMSE
RMSE dihitung dengan rumus berikut:

\[
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
\]

di mana:
- **n** adalah jumlah sampel,
- **yᵢ** adalah nilai aktual,
- **ŷᵢ** adalah nilai prediksi dari model,
- **(yᵢ - ŷᵢ)²** adalah error kuadrat dari setiap sampel.

## 🔢 Cara Perhitungan RMSE
1. Hitung selisih antara nilai aktual (**yᵢ**) dan nilai prediksi (**ŷᵢ**).
2. Kuadratkan selisih tersebut.
3. Hitung rata-rata dari semua error kuadrat (Mean Squared Error - MSE).
4. Ambil akar kuadrat dari MSE untuk mendapatkan RMSE.

## 📊 Interpretasi RMSE
✅ **RMSE kecil** → Model memiliki prediksi yang lebih akurat.  
❌ **RMSE besar** → Model memiliki kesalahan prediksi yang lebih tinggi.  

Keunggulan RMSE dibandingkan MSE:
- RMSE memiliki **satuan yang sama dengan target**, sehingga lebih mudah dipahami.
- Namun, RMSE **sensitif terhadap outlier**, karena error kuadrat diperbesar sebelum dihitung rata-ratanya.

## 📌 Perbedaan RMSE dan MSE
| Metrik | Formula | Interpretasi |
|--------|---------|--------------|
| **MSE** | \( \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \) | Rata-rata error kuadrat (bernilai besar karena dikuadratkan). |
| **RMSE** | \( \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} \) | Akar dari MSE, memiliki satuan yang sama dengan target sehingga lebih mudah dipahami. |

---


      



    
![png](gambar_files/gambar_116_0.png)
    



    
![png](gambar_files/gambar_116_1.png)
    


Grafik pertama menunjukkan RMSE untuk model berbasis jam ```(hour-based)```, sedangkan grafik kedua menunjukkan RMSE untuk model berbasis rating ```(rating-based)```.

1. Tren pada kedua grafik menunjukkan bahwa RMSE pada data latih (train) menurun seiring bertambahnya epoch, yang menandakan model semakin baik dalam menyesuaikan data latih.
2. RMSE pada data validasi (val) juga menurun tetapi lebih lambat dan cenderung stabil setelah beberapa epoch.
3. Model berbasis jam memiliki RMSE validasi yang lebih tinggi dibandingkan model berbasis rating, yang mungkin menunjukkan bahwa model rating-based lebih baik dalam melakukan generalisasi terhadap data baru.
4. Dari kedua grafik ini, tidak terlihat overfitting yang signifikan, tetapi jika RMSE validasi tidak turun lebih jauh, maka model mungkin sudah mencapai batas optimalnya.

## **References**

[[1](https://eraspace.com/artikel/post/sejumlah-perbandingan-steam-dan-epic-games-mana-yang-terbaik)]  Sejumlah perbandingan steam dan epic games mana yang terbaik
  
[[2](https://www.semanticscholar.org/paper/Implementasi-Chatbot-Telegram-untuk-Meningkatkan-Yuwan-Soelistijadi/fe61864360798ba62bada1ea70b790a0bd6e5f69)] Yuwan, Ridho Pangestu et al. “Implementasi Chatbot Telegram untuk Meningkatkan Kualitas Layanan Jaringan Internet Pada Layanan ICONNET Menggunakan Penerapan Metode Action Research (AR).” Jurnal JTIK (Jurnal Teknologi Informasi dan Komunikasi) (2024): n. pag. 

[[3](https://www.semanticscholar.org/paper/Analisis-Penerapan-UI-UX-Dalam-Meningkatkan-Pada-Syafei-Hidayatullah/ca7cfbaa16740716159ba1660203ce012d26ba16)] Syafei, Tubagus Fandi Maulana and Azka Fariz Hidayatullah. “Analisis Penerapan UI/UX Dalam Meningkatkan Pengalaman Pengguna Pada Sistem Reservasi Amadeus.” JUSTINFO | Jurnal Sistem Informasi dan Teknologi Informasi (2023): n. pag. 

[[4](https://www.semanticscholar.org/paper/Implementasi-Chatbot-Telegram-untuk-Meningkatkan-Yuwan-Soelistijadi/fe61864360798ba62bada1ea70b790a0bd6e5f69)] Yuwan, Ridho Pangestu et al. “Implementasi Chatbot Telegram untuk Meningkatkan Kualitas Layanan Jaringan Internet Pada Layanan ICONNET Menggunakan Penerapan Metode Action Research (AR).” Jurnal JTIK (Jurnal Teknologi Informasi dan Komunikasi) (2024): n. pag. 

[[5](https://www.semanticscholar.org/paper/Analisis-Penerapan-UI-UX-Dalam-Meningkatkan-Pada-Syafei-Hidayatullah/ca7cfbaa16740716159ba1660203ce012d26ba16)] Syafei, Tubagus Fandi Maulana and Azka Fariz Hidayatullah. “Analisis Penerapan UI/UX Dalam Meningkatkan Pengalaman Pengguna Pada Sistem Reservasi Amadeus.” JUSTINFO | Jurnal Sistem Informasi dan Teknologi Informasi (2023): n. pag.  

[[6](https://keras.io/examples/structured_data/collaborative_filtering_movielens/)] Penyaringan Kolaboratif untuk Rekomendasi Film
