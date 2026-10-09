# Data dictionary

The source workbook is `rice new one.xlsx`. It contains 149 seasonal rows spanning Yala 1950 to Yala 2024.

| Column | Meaning | Used in reference backtest? |
|---|---|---|
| `Year_New` | Calendar year associated with season | For labels/reporting only |
| `Season` | `Yala` or `Maha` | Yes; known future calendar feature |
| `Sown (*000  Acres)` | Sown area, in thousand acres | No; actual values may not be known at issue time |
| `Harvested (*000  Acres)` | Harvested area, in thousand acres | No; realized value is not available for a pre-season forecast |
| `GDP B$` | GDP as supplied in the workbook; source/price basis should be verified | No |
| `Inflation(%)` | Inflation rate as supplied in the workbook | No |
| `Rainfall(mm)` | Rainfall in millimetres; period aggregation should be verified against crop calendar | No; realized future-period weather is not available ex ante |
| `Temperature(°C)` | Temperature in degrees Celsius | No; realized future-period weather is not available ex ante |
| `Production (*000  Mt.)` | Rice production in thousand metric tonnes | Yes; target |

Before adding exogenous variables, document source URLs, retrieval dates, units, revisions, seasonal aggregation rules, and publication lag. If using realized in-season weather or acreage, describe the task as a nowcast/post-season estimate rather than an advance forecast.
