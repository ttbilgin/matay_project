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
