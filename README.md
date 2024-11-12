# matay_project

projeyi çalıştırmak için train ve test datasetlerini Colab'a upload edin ve MatayV2.ipynb dosyasındaki kodları çalıştırın

## Akış Diyagramı

```mermaid
flowchart TB
    subgraph Data_Preparation["Veri Hazırlama"]
        A[CSV Veri Seti] --> B[Zaman Damgası Düzenleme]
        B --> C[Etiket Dönüştürme<br>-99 -> 1, 0 -> 0]
        C --> D[Özellik Seçimi<br>15 Ana Özellik]
    end

    subgraph Feature_Engineering["Özellik Mühendisliği"]
        D --> E1[Hareketli Ortalama<br>Window=5]
        D --> E2[Standart Sapma<br>Window=5]
        D --> E3[Değişim Oranları]
        D --> E4[Hareketli Varyans<br>Window=5]
        D --> E5[Hareketli Medyan<br>Window=5]
    end

    subgraph Data_Processing["Veri İşleme"]
        E1 & E2 & E3 & E4 & E5 --> F[Missing Value İşleme<br>ffill & bfill]
        F --> G[Veri Normalizasyonu<br>RobustScaler]
        G --> H[Sekans Oluşturma<br>Sequence Length=20]
    end

    subgraph Model_Architecture["Model Mimarisi"]
        H --> I1[1D Konvolüsyon<br>32 Filters]
        I1 --> I2[1D Konvolüsyon<br>64 Filters]
        I2 --> J1[Bidirectional LSTM<br>64 Units]
        J1 --> J2[Bidirectional LSTM<br>32 Units]
        J2 --> K1[Dense Layer<br>64 Units]
        K1 --> K2[Dense Layer<br>32 Units]
        K2 --> L[Output Layer<br>1 Unit, Sigmoid]
    end

    subgraph Training["Model Eğitimi"]
        L --> M1[Custom Loss Function]
        L --> M2[Adam Optimizer]
        L --> M3[Callbacks<br>EarlyStopping & ReduceLROnPlateau]
        M1 & M2 & M3 --> N[Model Training]
    end

    subgraph Evaluation["Model Değerlendirme"]
        N --> O1[Classification Report]
        N --> O2[Confusion Matrix]
        N --> O3[PR & ROC Curves]
        O1 & O2 & O3 --> P[Final Model]
    end

```
## NN Architecture

```mermaid
graph TD
    subgraph Input["Input Layer"]
        A["Input Sequence<br>(20 timesteps × N features)"]
    end

    subgraph CNN["Convolutional Layers"]
        B1["Conv1D Layer 1<br>32 filters, kernel=3<br>+ LayerNorm<br>+ Dropout(0.2)"]
        B2["Conv1D Layer 2<br>64 filters, kernel=3<br>+ LayerNorm<br>+ Dropout(0.2)"]
    end

    subgraph LSTM["Bidirectional LSTM Layers"]
        C1["Bidirectional LSTM 1<br>64 units<br>+ LayerNorm<br>+ Dropout(0.3)"]
        C2["Bidirectional LSTM 2<br>32 units<br>+ LayerNorm<br>+ Dropout(0.3)"]
    end

    subgraph Dense["Dense Layers"]
        D1["Dense Layer 1<br>64 units<br>ReLU<br>+ LayerNorm<br>+ Dropout(0.2)"]
        D2["Dense Layer 2<br>32 units<br>ReLU<br>+ LayerNorm<br>+ Dropout(0.2)"]
    end

    subgraph Output["Output Layer"]
        E["Dense Layer<br>1 unit<br>Sigmoid"]
    end

    A --> B1
    B1 --> B2
    B2 --> C1
    C1 --> C2
    C2 --> D1
    D1 --> D2
    D2 --> E

    style A fill:#f9f,stroke:#333,stroke-width:4px
    style E fill:#f96,stroke:#333,stroke-width:4px
    style B1 fill:#bbf,stroke:#333,stroke-width:2px
    style B2 fill:#bbf,stroke:#333,stroke-width:2px
    style C1 fill:#bfb,stroke:#333,stroke-width:2px
    style C2 fill:#bfb,stroke:#333,stroke-width:2px
    style D1 fill:#fbf,stroke:#333,stroke-width:2px
    style D2 fill:#fbf,stroke:#333,stroke-width:2px
```
