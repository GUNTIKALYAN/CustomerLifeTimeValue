# CLTV Power BI Dashboard — Build Guide

## 1. Generate the dataset
```powershell
python -m src.export_powerbi
```
Output: `powerbi_cltv_data.csv` (59,595 customers with `predicted_cltv` + segment).

## 2. Import into Power BI Desktop
1. Open Power BI Desktop → **Get Data > Text/CSV** → select `powerbi_cltv_data.csv` → Load.
2. (Optional but recommended) Open **Power Query** → paste `power_query_cltv.m` to clean + type columns.

## 3. Add DAX measures
Modeling > New Measure → paste each block from `dax_measures.txt`.

## 4. Recommended visuals
| Page | Visual | Field |
|---|---|---|
| Overview | Cards | `Total CLTV`, `Total Customers`, `Avg CLTV` |
| Segments | Donut chart | Legend = `cltv_segment`, Values = `Total CLTV` |
| Geography | Bar chart | Axis = `area_label`, Values = `Total CLTV` |
| Policy value | Stacked bar | Axis = `type_of_policy`, Legend = `cltv_segment`, Values = `Total CLTV` |
| Demographics | Matrix | Rows = `qualification`, Columns = `cltv_segment`, Values = `Total Customers` |
| Loyalty | Scatter | X = `vintage`, Y = `predicted_cltv`, Size = `Total CLTV` |

## 5. Business story to tell (interview)
- "The model predicts **~₹77K average CLTV** across 60K customers — the **Premium segment (top 25%)** drives the majority of total value."
- "Premium CLTV Share % vs Premium Customer Share % shows **value concentration** → target retention spend there."
- "Scatter of vintage vs predicted CLTV shows **loyal customers are more valuable** → focus on retention."

## 6. Save & export
- Save as `.pbix`.
- **File > Publish** to Power BI Service for sharing.
- Screenshot the dashboard → add to your resume/portfolio (defends the "Power BI" bullet).
