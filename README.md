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




# Deep Learning Architecture for Welding Fault Detection

## Introduction
The presented code implements a sophisticated deep learning architecture designed for predictive fault detection in robotic welding processes. The model utilizes a hybrid approach combining Convolutional Neural Networks (CNNs) and Bidirectional Long Short-Term Memory (BiLSTM) networks to capture both spatial and temporal patterns in sensor data.

## Data Preprocessing and Feature Engineering
The raw sensor data undergoes extensive preprocessing to enhance the model's learning capabilities. The temporal data is structured with timestamps and binary fault labels, where -99 indicates a fault occurrence (converted to 1) and 0 represents normal operation. The feature space comprises 15 primary sensor measurements including current, voltage, and power metrics.

Feature engineering involves the creation of rolling statistics with a window size of 5, including:
- Moving averages: $MA(t) = \frac{1}{w}\sum_{i=t-w+1}^{t} x_i$
- Rolling standard deviation: $$\sigma(t) = \sqrt{\frac{1}{w-1}\sum_{i=t-w+1}^{t} (x_i - \mu)^2}$$
- Rate of change: $$\Delta x(t) = x_t - x_{t-1}$$
- Moving variance and median calculations

Data normalization is performed using RobustScaler to handle outliers effectively, following the transformation:
$$x_{scaled} = \frac{x - Q_1}{Q_3 - Q_1}$$
where $$Q_1$$ and $$Q_3$$ represent the first and third quartiles respectively.

## Sequence Generation and Processing
Time series sequences are generated with a length of 20 timesteps, incorporating a pre-fault window of 150 timesteps for fault cases. The sequence generation process implements an advanced sampling strategy to address class imbalance, maintaining a ratio of 3:1 between normal and fault sequences.

## Neural Network Architecture
The model architecture consists of multiple specialized layers:

### Convolutional Layers
Two 1D convolutional layers process the input sequences:
$$Conv1D_1(x) = \sigma(W_1 * x + b_1)$$
$$Conv1D_2(x) = \sigma(W_2 * Conv1D_1(x) + b_2)$$

where * denotes the convolution operation and $$\sigma$$ represents the ReLU activation function:
$$\sigma(x) = max(0, x)$$

### Bidirectional LSTM Layers
The BiLSTM layers process the sequence in both forward and backward directions:
$$\vec{h_t} = LSTM_{forward}(x_t, \vec{h_{t-1}})$$
$$\overleftarrow{h_t} = LSTM_{backward}(x_t, \overleftarrow{h_{t+1}})$$

The final output combines both directions:
$$h_t = [\vec{h_t}, \overleftarrow{h_t}]$$

### Dense Layers and Regularization
Each dense layer applies the transformation:
$$z_l = W_l h_{l-1} + b_l$$
$$h_l = \sigma(LayerNorm(z_l))$$

Layer normalization is applied as:
$$LayerNorm(x) = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

Dropout is implemented with probabilities of 0.2 and 0.3 for different layers:
$$h_{dropped} = h \odot Bernoulli(p)$$

## Loss Function and Training
The custom loss function incorporates class weights and confidence penalties:
$$L(y, \hat{y}) = BCE(y, \hat{y}) \cdot w(y) \cdot c(\|y - \hat{y}\|)$$

where $$BCE$$ is binary cross-entropy, $$w(y)$$ is the class weight function, and $$c(x)$$ is the confidence penalty function.

The Adam optimizer is utilized with the following parameters:
- Learning rate: $$\alpha = 0.001$$
- Exponential decay rates: $$\beta_1 = 0.9$$, $$\beta_2 = 0.999$$
- Epsilon: $$\epsilon = 10^{-7}$$

## Model Evaluation
Performance evaluation includes multiple metrics:
- Precision-Recall curve and AUC
- ROC curve and AUC
- Confusion matrix analysis
- Classification report with precision, recall, and F1-score

The model implements early stopping and learning rate reduction strategies to prevent overfitting and optimize convergence. The validation process uses a split ratio of 0.2, with stratification to maintain class distribution.






