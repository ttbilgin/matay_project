# Comprehensive Dataset Description for Robotic Welding Process

## Temporal Information

| Feature | Data Type | Format/Range | Description |
|---------|-----------|--------------|-------------|
| timestamp | datetime64[ns] | ISO8601 UTC | High-precision timestamp capturing the exact moment of measurement, essential for temporal analysis and sequence modeling |

## Current Measurements

| Feature | Data Type | Range | Unit | Purpose & Description |
|---------|-----------|--------|------|---------------------|
| Current L1 | float64 | 0.0 - 5.0 | A | Line 1 current measurement. Monitors the current flow in phase 1 of the welding circuit. Critical for detecting phase imbalances and overload conditions |
| Current L2 | float64 | 0.0 - 5.0 | A | Line 2 current measurement. Monitors phase 2 current flow. Important for three-phase balance analysis |
| Current L3 | float64 | 0.0 - 5.0 | A | Line 3 current measurement. Monitors phase 3 current flow. Completes the three-phase current monitoring system |
| Current Unbalance | float64 | 0.0 - 100.0 | % | Percentage difference between phase currents. Calculated as: $$\frac{max(|I_1-I_2|,|I_2-I_3|,|I_3-I_1|)}{I_{avg}} \times 100$$ Critical for detecting asymmetric loading |
| Neutral Current | float64 | 0.0 - 2.0 | A | Current in the neutral conductor. Should be minimal in balanced systems. High values indicate potential phase imbalance or harmonics |

## Voltage Measurements

| Feature | Data Type | Range | Unit | Purpose & Description |
|---------|-----------|--------|------|---------------------|
| Voltage L1-L2 | float64 | 380.0 - 420.0 | V | Line voltage between phases 1 and 2. Critical for monitoring power quality and phase-to-phase voltage stability |
| Voltage L2-L3 | float64 | 380.0 - 420.0 | V | Line voltage between phases 2 and 3. Part of complete three-phase voltage monitoring |
| Voltage L3-L1 | float64 | 380.0 - 420.0 | V | Line voltage between phases 3 and 1. Completes the line-to-line voltage measurement set |
| Voltage L1-N | float64 | 220.0 - 240.0 | V | Phase 1 to neutral voltage. Important for single-phase load analysis |
| Voltage L2-N | float64 | 220.0 - 240.0 | V | Phase 2 to neutral voltage. Monitors phase 2 voltage relative to neutral |
| Voltage L3-N | float64 | 220.0 - 240.0 | V | Phase 3 to neutral voltage. Completes phase-to-neutral voltage monitoring |
| Voltage Unbalance | float64 | 0.0 - 5.0 | % | Percentage of voltage imbalance between phases. Calculated as: $$\frac{max(|V_1-V_2|,|V_2-V_3|,|V_3-V_1|)}{V_{avg}} \times 100$$ |

## Power Measurements

| Feature | Data Type | Range | Unit | Purpose & Description |
|---------|-----------|--------|------|---------------------|
| ΣActive Power | float64 | 0.0 - 100.0 | kW | Total real power consumption. Represents actual work being done. Calculated as: $$P = \sum_{i=1}^3 V_i I_i \cos(\phi_i)$$ |
| ΣApparent Power | float64 | 0.0 - 120.0 | kVA | Total apparent power. Vector sum of active and reactive power. Calculated as: $$S = \sum_{i=1}^3 V_i I_i$$ |
| ΣCos Phi | float64 | -1.0 to 1.0 | - | Power factor angle cosine. Indicates the phase angle between voltage and current. Optimal value is close to ±1 |
| ΣPower Factor | float64 | -1.0 to 1.0 | - | Ratio of active to apparent power. Calculated as: $$PF = \frac{P}{S}$$ Indicates energy efficiency |

## Energy Measurements

| Feature | Data Type | Range | Unit | Purpose & Description |
|---------|-----------|--------|------|---------------------|
| Total Consumed Energy | float64 | ≥ 0.0 | kWh | Cumulative active energy consumption. Integral of active power over time: $$E = \int P dt$$ |
| Total Consumed Reactive Energy | float64 | ≥ 0.0 | kVArh | Cumulative reactive energy. Integral of reactive power over time: $$E_r = \int Q dt$$ |

## System Parameters

| Feature | Data Type | Range | Unit | Purpose & Description |
|---------|-----------|--------|------|---------------------|
| Measured Frequency | float64 | 49.5 - 50.5 | Hz | Power system frequency. Critical for system stability monitoring. Should remain close to nominal value (50 Hz) |
| Fault | int8 | {0, -99} | - | Fault indicator. Values:<br>0: Normal operation<br>-99: Fault detected |

## Derived Metrics and Calculations

### Power Quality Indicators

The dataset enables calculation of several important power quality metrics:

Total Harmonic Distortion (THD):
$$THD = \sqrt{\sum_{n=2}^{\infty} \left(\frac{I_n}{I_1}\right)^2} \times 100\%$$

Phase Unbalance Factor:
$$\text{Unbalance}_{\text{phase}} = \frac{\text{max deviation from average}}{\text{average}} \times 100\%$$

### Energy Efficiency Metrics

Overall system efficiency can be monitored through:

Power Factor Quality:
$$PF_{\text{quality}} = \frac{\text{Active Power}}{\sqrt{(\text{Active Power})^2 + (\text{Reactive Power})^2}}$$

## Data Quality Considerations

- Sampling Rate: 1 millisecond resolution
- Missing Values: Handled through forward and backward filling
- Outlier Detection: Based on physical constraints and statistical analysis
- Signal Noise: Typical industrial environment noise levels present

## Operational Context

The dataset captures the complete electrical characteristics of a robotic welding process, enabling:
- Real-time monitoring of welding quality
- Predictive maintenance through pattern analysis
- Power quality assessment
- Energy efficiency optimization
- Fault prediction and detection

Each parameter has been carefully selected to provide comprehensive coverage of the welding process dynamics while maintaining practical relevance for industrial applications.


# Temporal Information Analysis

## Training Set Specifications

| Feature | Data Type | Format | Range | Resolution | No of Samples |
|---------|-----------|--------|--------|------------|-------------|
| timestamp | datetime64[ns] | ISO8601 UTC<br>YYYY-MM-DD HH:mm:ss.fff+00:00 | Start: 2024-09-05 03:01:51.554<br>End: 2024-10-16 05:35:50.81 | 1 millisecond | 4660000 |

## Test Set Specifications

| Feature | Data Type | Format | Range | Resolution | No of Samples |
|---------|-----------|--------|--------|------------|-------------|
| timestamp | datetime64[ns] | ISO8601 UTC<br>YYYY-MM-DD HH:mm:ss.fff+00:00 | Start: 2024-10-22 02:54:30.877<br>End: 2024-10-23 08:22:11.987 | 1 millisecond | 167000 |

## Temporal Characteristics

### Sampling Pattern Analysis
- **Minimum Time Delta**: 1 millisecond
- **Typical Time Delta**: Variable, with following patterns:
  - Sequential measurements: 1ms intervals
  - Regular sampling gaps: ~150-200ms intervals
  - Process cycle gaps: ~10s intervals

### Time Series Structure
```
Timeline visualization:
12:41:26 [.......] 12:41:38 [.......] 12:41:49 [.......] 12:42:01 [.......] 12:42:12 [.......] 12:42:23
     |_1ms steps_|      |_1ms steps_|      |_1ms steps_|      |_1ms steps_|      |_1ms steps_|
```

### Temporal Resolution Details
- **Microsecond Precision**: Maintained in storage but rounded to millisecond in practice
- **Timezone**: UTC (indicated by +00:00 suffix)
- **Time Format Compliance**: ISO8601 standard with extended precision

### Sampling Characteristics
- **Regular Intervals**: Primary sampling at 1ms during active measurement periods
- **Measurement Bursts**: Groups of consecutive 1ms samples
- **Inter-burst Intervals**: Larger gaps between measurement bursts
- **Process Cycles**: Approximately 10-second intervals between major measurement sequences

This temporal structure reflects a high-frequency sampling system designed to capture rapid changes in the welding process while maintaining practical data storage and processing requirements. The timestamp format and precision are chosen to ensure accurate temporal alignment of measurements and enable detailed time-series analysis of the welding process dynamics.
